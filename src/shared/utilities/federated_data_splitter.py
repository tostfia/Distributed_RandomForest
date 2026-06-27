import os
import numpy as np
import boto3
from botocore.exceptions import ClientError
from src.shared.utilities.datasplitter import StratifiedDataSplitter

class FederatedDataSplitter:

    def __init__(self, target_column="Label", test_size=0.20, random_state=123):
        self.target_column = target_column
        self.random_state = random_state
        self.central_splitter = StratifiedDataSplitter(target_column=target_column, test_size=test_size, random_state=random_state)

    def split_and_shard(self, loader, num_workers: int, environment: str = "local", bucket_name: str = None):
        """
        Esegue lo sharding orizzontale del dataset estratto dal loader passatogli.
        In 'local' scrive le cartelle sul File System ospite.
        In 'aws' effettua il caricamento dei singoli frammenti direttamente in un bucket S3.
        """
        print(f"\n[FederatedDataSplitter] Avvio ripartizione per {num_workers} nodi federati (Ambiente: {environment.upper()})...")
        
        # Invocazione del metodo del loader
        df = loader.load()
        
        if df is None or df.empty:
            raise ValueError("[FederatedDataSplitter ERRORE] Il DataFrame caricato è vuoto o non valido.")

        # 1. Macro-Split Stratificato (Train globale / Test globale)
        train_df, test_df = self.central_splitter.split(df)
        
        # 2. Mescolamento casuale controllato
        train_df = train_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # 3. Suddivisione orizzontale equa dei chunk per i worker
        # 3. Suddivisione orizzontale equa dei chunk per i worker
        chunk_size_train = int(np.ceil(len(train_df) / num_workers))
        chunk_size_test = int(np.ceil(len(test_df) / num_workers))

        # Generazione controllata basata esattamente sul numero di worker
        train_shards = []
        test_shards = []
        for idx in range(num_workers):
            start_tr = idx * chunk_size_train
            end_tr = min(start_tr + chunk_size_train, len(train_df))
            train_shards.append(train_df.iloc[start_tr:end_tr])

            start_te = idx * chunk_size_test
            end_te = min(start_te + chunk_size_test, len(test_df))
            test_shards.append(test_df.iloc[start_te:end_te])

        

        if environment == "local":
            base_cache_dir = "./workers_cache"
            # --- SCENARIO LOCALE / DOCKER (File System Condiviso) ---
            for i in range(num_workers):
                worker_id = f"Worker-Locale-0{i+1}" if i < 9 else f"Worker-Locale-{i+1}"
                cache_dir = os.path.join(base_cache_dir, worker_id)
                os.makedirs(cache_dir, exist_ok=True)
                train_shards[i].to_csv(os.path.join(cache_dir, "train_shard.csv"), index=False)
                test_shards[i].to_csv(os.path.join(cache_dir, "test_shard.csv"), index=False)
                print(f"[Splitter LOCALE] Scritto shard per {worker_id} nella directory: {cache_dir}")
                
        elif environment == "aws":
            # --- SCENARIO CLOUD AWS (Storage ad Oggetti S3) ---
            if not bucket_name:
                bucket_name = "my-cluster-datasets-bucket"
                
            s3_client = boto3.client('s3')
            print(f"[Splitter AWS] Connessione a S3 effettuata. Upload degli shard nel bucket '{bucket_name}'...")
            
            for i in range(num_workers):
                worker_id = f"worker_{i+1}"
                
                # Convertiamo in stringa CSV direttamente in RAM senza toccare l'hard disk dell'EC2 Master
                train_csv = train_shards[i].to_csv(index=False)
                # Salviamo anche lo shard di test se serve per le validazioni locali dei worker
                test_csv = test_shards[i].to_csv(index=False)
                
                train_key = f"federated_shards/{worker_id}/train_shard.csv"
                test_key = f"federated_shards/{worker_id}/test_shard.csv"
                
                try:
                    s3_client.put_object(Bucket=bucket_name, Key=train_key, Body=train_csv)
                    s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=test_csv)
                    print(f"[Splitter AWS] Caricato Shard con successo -> Key S3: {train_key}")
                except ClientError as e:
                    print(f"[Splitter AWS ERRORE] Fallimento dell'upload per il {worker_id}: {e}")
                    raise e