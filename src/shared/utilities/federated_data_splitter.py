import os
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from src.shared.utilities.datasplitter import StratifiedDataSplitter

BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")

VALID_PARTITION_STRATEGIES = ("iid","by_day")


class FederatedDataSplitter:

    def __init__(self, target_column="Label", test_size=0.20, random_state=123):
        self.target_column = target_column
        self.random_state = random_state
        self.central_splitter = StratifiedDataSplitter(target_column=target_column, test_size=test_size, random_state=random_state)

    def split_and_shard(self, loader, num_workers: int, environment: str = "local", bucket_name: str = None,
                         partition_strategy: str = "iid", day_column: str = None):
        """
        Esegue lo sharding orizzontale del dataset estratto dal loader passatogli.
        In 'local' scrive le cartelle sul File System ospite.
        In 'aws' effettua il caricamento dei singoli frammenti direttamente in un bucket S3.

        partition_strategy:
            - "iid" (default, comportamento storico invariato): mescolamento globale
              casuale e chunk di dimensione uguale per ciascun worker.
            - "by_day": partizionamento "naturale" per file/giorno di origine, a zero
              parametri. Richiede che il DataFrame caricato dal loader contenga una
              colonna che identifica il giorno/file sorgente, il cui nome va passato
              in 'day_column' (es. day_column="source_day").


        day_column: nome colonna usato solo con partition_strategy="by_day".
        """
        if partition_strategy not in VALID_PARTITION_STRATEGIES:
            raise ValueError(
                f"[FederatedDataSplitter ERRORE] partition_strategy sconosciuta: '{partition_strategy}'. "
                f"Valori validi: {VALID_PARTITION_STRATEGIES}."
            )

        print(f"\n[FederatedDataSplitter] Avvio ripartizione per {num_workers} nodi federati "
              f"(Ambiente: {environment.upper()}, strategia: {partition_strategy.upper()}")
        
        # Invocazione del metodo del loader
       
        df = loader.load()

        if df is None or df.empty:
            raise ValueError("[FederatedDataSplitter ERRORE] Il DataFrame caricato è vuoto o non valido.")

        # 1. Macro-Split Stratificato (Train globale / Test globale)
        train_df, test_df = self.central_splitter.split(df)

        # 2. Partizionamento orizzontale tra i worker secondo la strategia scelta
        if partition_strategy == "iid":
            # Mescolamento casuale controllato: comportamento storico invariato.
            train_df = train_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            train_shards = self._shard_iid(train_df, num_workers)
            test_shards = self._shard_iid(test_df, num_workers)
        else:  # "by_day"
            if not day_column or day_column not in train_df.columns:
                raise ValueError(
                    "[FederatedDataSplitter ERRORE] partition_strategy='by_day' richiede un 'day_column' "
                    f"valido presente nel DataFrame caricato (ricevuto: {day_column!r})."
                )
            train_shards = self._shard_by_day(train_df, num_workers, day_column)
            test_shards = self._shard_by_day(test_df, num_workers, day_column)

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
                bucket_name = BUCKET_NAME

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

    # ------------------------------------------------------------------
    # Strategie di partizionamento
    # ------------------------------------------------------------------

    def _shard_iid(self, df, num_workers: int):
        """
        Suddivisione orizzontale equa e contigua dei chunk per i worker (comportamento
        storico). Va chiamata su un DataFrame già mescolato casualmente a monte, così
        ogni worker riceve una porzione IID del dataset globale.
        """
        chunk_size = int(np.ceil(len(df) / num_workers))
        shards = []
        for idx in range(num_workers):
            start = idx * chunk_size
            end = min(start + chunk_size, len(df))
            shards.append(df.iloc[start:end])
        return shards

    def _shard_by_day(self, df, num_workers: int, day_column: str):
        """
        Partizionamento "naturale" per file/giorno di origine, a zero parametri.
        Ogni valore distinto di day_column viene assegnato per intero a un worker
        (round-robin), così l'eterogeneità già presente nei dati grezzi (es. i CSV
        giornalieri di CIC-IDS2018, ciascuno con uno scenario di attacco dominante
        diverso) non viene distrutta dal mescolamento globale.
        """
        days = sorted(df[day_column].unique(), key=str)
        if len(days) < num_workers:
            print(f"[FederatedDataSplitter] [ATTENZIONE] Solo {len(days)} valori distinti di "
                  f"'{day_column}' disponibili per {num_workers} worker: alcuni worker "
                  f"riceveranno uno shard vuoto.")

        worker_frames = [[] for _ in range(num_workers)]
        for i, day in enumerate(days):
            worker_frames[i % num_workers].append(df[df[day_column] == day])

        shards = []
        for w in range(num_workers):
            if worker_frames[w]:
                shards.append(pd.concat(worker_frames[w], ignore_index=True))
            else:
                shards.append(df.iloc[0:0].reset_index(drop=True))
        return shards