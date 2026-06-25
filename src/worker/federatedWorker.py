import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.worker.BaseWorker import BaseWorker
from src.shared.factory import DatasetDAOFactory  # Utilizziamo il DAO centralizzato


class FederatedWorker(BaseWorker):
    """Worker per la gestione dell'addestramento in modalità federata.

    Interpreta le stringhe sintetiche oppure scarica lo shard reale assegnato
    dall'Orchestratore salvandolo nella cache del disco rigido locale (EBS su AWS).
    """

    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        tree_class_reference: type,
        target_column: str = "Label",
        max_samples: float = None,
        bootstrap: bool = True,
        tree_type: str = "classifier"
    ):
        super().__init__(
            worker_name=worker_name,
            queue_name=queue_name,
            tree_class_reference=tree_class_reference,
            max_samples=max_samples,
            bootstrap=bootstrap,
        )
        self.target_column = target_column
        self.tree_type = tree_type
        
        # Gestione asimmetrica della cache locale in base all'ambiente
        if self.environment == "aws":
            self.local_cache_dir = f"/tmp/{worker_name}_cache"
        else:
            self.local_cache_dir = f"./{worker_name}_cache"
            
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Attributi per preservare il testing set locale per l'inferenza federata
        self.X_test = None
        self.y_test = None
        self.local_sample_count = 0

        self.worker_index = 0
        for char in worker_name.split("-"):
            if char.isdigit():
                self.worker_index = int(char)
                break

        print(
            f"[FederatedWorker] Inizializzato in ambiente: {self.environment.upper()} — "
            f"Directory Cache locale: {self.local_cache_dir} — Target: {self.target_column}"
        )

    def is_regression(self) -> bool:
        return self.tree_type == "regressor"

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        """Carica i dati per l'addestramento locale e conserva una quota di split

        per la successiva validazione/inferenza distribuita.

        Args:
            source_info (str): Può essere 'NATIVE_PARTITIONED|i' o un path/URL
              S3 fornito dall'Orchestratore.
        """
        X_raw, y_raw = None, None
        
        # --- OPZIONE A: GENERAZIONE SINTETICA LOCALE AUTONOMA ---
        if source_info.startswith("NATIVE_PARTITIONED"):
            idx_da_stringa = self.worker_index
            if "|" in source_info:
                try:
                    idx_da_stringa = int(source_info.split("|")[1])
                except ValueError:
                    pass

            seed_locale = 123 + idx_da_stringa
            print(f"[FederatedWorker] Generazione sintetica in RAM (Index: {idx_da_stringa}, Seed: {seed_locale})...")
            
            n_samples = 25000  
            n_features = 20
            
            if self.is_regression():
                X_raw, y_raw = make_regression(
                    n_samples=n_samples, n_features=n_features, noise=0.1, random_state=seed_locale
                )
                y_raw = y_raw.astype(np.float64)
            else:
                X_raw, y_raw = make_classification(
                    n_samples=n_samples, n_features=n_features, n_informative=15, n_classes=2, random_state=seed_locale
                )
                y_raw = y_raw.astype(np.int64)

            # Per il dataset sintetico facciamo lo split 80/20 standard in RAM al volo
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y_raw, test_size=0.20, random_state=42, stratify=None if self.is_regression() else y_raw
            )
            self.X_test = X_test
            self.y_test = y_test

        # --- OPZIONE B: DATASET REALE (Lettura dallo Shard distribuito dall'Orchestratore) ---
        else:
            print(f"[FederatedWorker] Ricevuto path sorgente dall'Orchestratore: {source_info}")
            
            # Utilizziamo il DAO Factory per scaricare e leggere lo shard (funziona sia per S3 che per locale)
            dao = DatasetDAOFactory.get_dao(self.environment)
            
            # Caso AWS Learner Lab: Se il path è su S3, lo scarichiamo localmente sul disco EBS (/tmp)
            # per simulare l'isolamento hardware ed evitare letture di rete continue.
            if source_info.startswith("s3://"):
                filename = os.path.basename(source_info)
                local_train_path = os.path.join(self.local_cache_dir, filename)
                local_test_path = os.path.join(self.local_cache_dir, filename.replace("train_", "test_"))
                
                print(f"[FederatedWorker] Cache Cloud: Download del proprio shard train locale su: {local_train_path}")
                # Scarichiamo il file di train se non già presente in cache
                if not os.path.exists(local_train_path):
                    df_train_cloud = dao.load_dataset(source_info)
                    df_train_cloud.to_csv(local_train_path, index=False)
                
                # Scarichiamo preventivamente il file di test associato per l'inferenza futura
                if not os.path.exists(local_test_path):
                    s3_test_url = source_info.replace("train_", "test_")
                    df_test_cloud = dao.load_dataset(s3_test_url)
                    df_test_cloud.to_csv(local_test_path, index=False)
                
                # Aggiorniamo i puntatori ai file locali sul file system (EBS)
                path_to_read_train = local_train_path
                path_to_read_test = local_test_path
            else:
                # Caso Locale: i file sono già stati partizionati dall'orchestratore nelle cartelle dei worker
                path_to_read_train = source_info
                path_to_read_test = source_info.replace("train_", "test_")

            # --- CARICAMENTO EFFETTIVO DEI COORTI DAL DISCO LOCALE ---
            print(f"[FederatedWorker] Caricamento train set da cache locale: {path_to_read_train}")
            df_train = pd.read_csv(path_to_read_train)
            X_train = df_train.drop(columns=[self.target_column]).to_numpy(dtype=np.float64)
            y_train = df_train[self.target_column].to_numpy(dtype=np.float64 if self.is_regression() else np.int64)

            print(f"[FederatedWorker] Caricamento e bloccaggio in RAM del test set da cache locale: {path_to_read_test}")
            df_test = pd.read_csv(path_to_read_test)
            self.X_test = df_test.drop(columns=[self.target_column]).to_numpy(dtype=np.float64)
            self.y_test = df_test[self.target_column].to_numpy(dtype=np.float64 if self.is_regression() else np.int64)

        self.local_sample_count = len(X_train)
        print(f"[FederatedWorker] [ETL OK] Record di addestramento pronti: {X_train.shape}")
        print(f"[FederatedWorker] [ETL OK] Record di validazione blindati in RAM: {self.X_test.shape}")
        
        return X_train, y_train

    def _get_tree_class(self) -> type:
        return self.tree_class_reference
    
    #override 
    def _get_task_storage_paths(self, source_info: str, base_seed: int, num_trees: int):
        """
        OVERRIDE CRITICO: Forza l'isolamento dei task su storage condiviso (S3/Locale)
        iniettando il nome del worker nel path per evitare sovrascritture tra nodi diversi.
        """
        filename = os.path.basename(source_info)
        job_id = filename.replace("train_", "").replace(".csv", "").replace("|", "_")

        local_dir = os.path.join("./.local_storage","trained_tasks")
        local_path = os.path.join(local_dir,f"task_{self.worker_name}_{job_id}_seed_{base_seed}_trees_{num_trees}.json")
        s3_bucket = os.environ.get("TRAINED_TREES_S3_BUCKET", "my-cluster-trained-trees-bucket")
        # Inseriamo una sottocartella specifica per il worker su S3
        s3_key = f"tasks/{job_id}/{self.worker_name}/task_seed_{base_seed}_trees_{num_trees}.pkl"
        
        return local_dir, local_path, s3_bucket, s3_key

    def exposed_get_local_y_test(self) -> bytes:
        """Metodo esposto tramite RPC per consentire all'Orchestratore di scaricare

        le etichette reali del testing set locale durante la computazione delle metriche.
        """
        if self.y_test is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self.y_test)
    
    def exposed_get_local_sample_count(self)-> int:
        """Metodo esposto tramite RPC per consentire all'Orchestratore di ottenere

        il numero di campioni locali per il calcolo ponderato delle metriche globali.
        """
        return self.local_sample_count