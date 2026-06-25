import os
import pickle
from botocore.exceptions import ClientError
import boto3
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

from Distributed_RandomForest.src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.worker.BaseWorker import BaseWorker
from src.shared.factory import DatasetDAOFactory 


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
        
        # Cache dello stato interno
        self._cached_job_id = None
        self._cached_X_train = None
        self._cached_y_train = None
        self._cached_X_test = None
        self._cached_y_test = None
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
    
    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference
   
    def _load_data(self, dataset_tag: str):
        """Implementazione obbligatoria per la classe base."""
        
        dao = DatasetDAOFactory.get_dao(self.environment)
        
        # Carica dai percorsi standard definiti dallo splitter
        train_path = os.path.join(self.local_cache_dir, "train_shard.csv")
        test_path = os.path.join(self.local_cache_dir, "test_shard.csv")
        
        train_df = dao.load_dataset(train_path)
        self.X_train = train_df.drop(columns=[self.target_column])
        self.y_train = train_df[self.target_column]
        
        test_df = dao.load_dataset(test_path)
        self.X_test = test_df.drop(columns=[self.target_column])
        self.y_test = test_df[self.target_column]
        return True

    def exposed_load_local_shard(self):
        """Metodo RPC per forzare il caricamento."""
        return self._load_data("real")
    
    def exposed_train_local_federated_forest(self, payload: dict) -> list:
        """Metodo esposto tramite RPC richiesto dall'Orchestratore per avviare l'addestramento.
        Chiama il caricamento dinamico dei dati e restituisce gli alberi locali in formato binario.
        """
        """Metodo RPC esposto all'Orchestratore per l'addestramento locale."""
        job_id = payload.get("job_id")
        dataset_type = payload.get("dataset_type", "real").strip().lower()
        n_estimators_local = int(payload.get("n_estimators_local", 1))
        worker_index = payload.get("worker_index", 1)
        hyperparameters = payload.get("hyperparameters", {})

        max_depth = hyperparameters.get("max_depth")
        base_seed = int(hyperparameters.get("random_state", 123))
        
        print(f"\n[{self.worker_name}] Ricevuto Task RPC Federato per Job {job_id[:8]}")

        # Caricamento e Preprocessing dei dati (eseguito solo se cambia il Job ID o la cache è vuota)
        if self._cached_job_id != job_id or self._cached_X_train is None:
            if dataset_type == "synthetic":
                self._load_synthetic_data(hyperparameters)
            else:
                # Passiamo l'intero payload per estrarre la lista globale delle feature sincronizzate
                self._load_and_preprocess_real_shard(worker_index, payload)
            self._cached_job_id = job_id

        # Configurazione dei seed atomici per i singoli alberi del chunk
        seeds_for_trees = [base_seed + i for i in range(n_estimators_local)]
        task_arguments = [
            (seed, max_depth, self.max_samples, self.bootstrap, self._get_tree_class())
            for seed in seeds_for_trees
        ]

        print(f"[{self.worker_name}] Avvio computazione parallela di {n_estimators_local} alberi...")
        trained_trees = self._execute_parallel_training(task_arguments, n_estimators_local)

        return trained_trees
    
    def _load_and_preprocess_real_shard(self, worker_index: int, payload: dict):
        """Scarica e processa lo shard reale garantendo l'allineamento delle feature."""
        train_filename = "train_shard.csv"
        test_filename = "test_shard.csv"
        local_train_path = os.path.join(self.local_cache_dir, train_filename)
        local_test_path = os.path.join(self.local_cache_dir, test_filename)

        # 1. GESTIONE AWS S3 (Se applicabile)
        if self.environment == "aws":
            bucket_name = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket")
            s3_train_key = f"federated_shards/worker_{worker_index}/{train_filename}"
            s3_test_key = f"federated_shards/worker_{worker_index}/{test_filename}"
            
            s3_client = boto3.client("s3")
            try:
                s3_client.download_file(bucket_name, s3_train_key, local_train_path)
                s3_client.download_file(bucket_name, s3_test_key, local_test_path)
            except ClientError as e:
                raise IOError(f"[{self.worker_name}] Errore download shard da S3: {e}")

        # 2. LETTURA SHARD GREZZI
        if not os.path.exists(local_train_path):
            raise FileNotFoundError(f"Shard non trovato in {local_train_path}")
            
        df_train_raw = pd.read_csv(local_train_path, low_memory=False)
        df_test_raw = pd.read_csv(local_test_path, low_memory=False)

        # 3. PIPELINE DI PREPROCESSING SPECULARE
        preprocessor = CICIDSPreprocessor(target_column=self.target_column)
        
        df_train_bin = preprocessor.binarize_target(df_train_raw)
        df_test_bin = preprocessor.binarize_target(df_test_raw)
        
        df_train_clean = preprocessor.process(df_train_bin)
        df_test_clean = preprocessor.process(df_test_bin)

        # 4. Allineamento Forzato delle Feature (Sincronizzazione tramite Master)
        # L'orchestratore includerà nel payload la lista esatta 'selected_features' decisa a monte
        selected_features = payload.get("selected_features", None)
        
        if selected_features is not None:
            print(f"[{self.worker_name}] Applicazione spazio feature sincronizzato dal Master ({len(selected_features)} colonne).")
            # Ci assicuriamo che il target rimanga presente prima di isolare le matrici
            features_to_keep = [col for col in selected_features if col != self.target_column]
            
            X_train_df = df_train_clean[features_to_keep]
            X_test_df = df_test_clean[features_to_keep]
        else:
            # Fallback di sicurezza se non passate (es. esecuzione isolata)
            print(f"[{self.worker_name}] [ATTENZIONE] Nessuna lista feature passata. Uso il set completo.")
            X_train_df = df_train_clean.drop(columns=[self.target_column])
            X_test_df = df_test_clean.drop(columns=[self.target_column])

        y_train_df = df_train_clean[self.target_column]
        y_test_df = df_test_clean[self.target_column]

        # 5. CONVERSIONE IN MATRICI NUMPY STABILI (Identica a CentralizedWorker)
        self._cached_X_train = X_train_df.to_numpy(dtype=np.float64)
        self._cached_X_test = X_test_df.to_numpy(dtype=np.float64)
        
        if self.is_regression():
            self._cached_y_train = y_train_df.to_numpy(dtype=np.float64)
            self._cached_y_test = y_test_df.to_numpy(dtype=np.float64)
        else:
            self._cached_y_train = y_train_df.to_numpy(dtype=np.int64)
            self._cached_y_test = y_test_df.to_numpy(dtype=np.int64)
        
        self.local_sample_count = len(self._cached_X_train)
        print(f"[{self.worker_name}] Shard caricato in cache. X_train Shape: {self._cached_X_train.shape}")

    def _load_synthetic_data(self, hyperparameters: dict):
        """Generazione di dati sintetici speculari in RAM."""
        seed = hyperparameters.get("random_state", 123)
        X, y = make_classification(n_samples=20000, n_features=20, random_state=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
        
        self._cached_X_train = X_tr.astype(np.float64)
        self._cached_X_test = X_te.astype(np.float64)
        
        if self.is_regression():
            self._cached_y_train = y_tr.astype(np.float64)
            self._cached_y_test = y_te.astype(np.float64)
        else:
            self._cached_y_train = y_tr.astype(np.int64)
            self._cached_y_test = y_te.astype(np.int64)
            
        self.local_sample_count = len(self._cached_X_train)

    def exposed_predict_subset_forest(self, serialized_trees: bytes, serialized_X_test: bytes = None) -> bytes:
        
        if self._cached_X_test is None:
            raise ValueError("Nessun dataset di test in cache.")
            
        unpacked_model = pickle.loads(serialized_trees)
        
        if isinstance(unpacked_model, list):
            if self.is_regression():
                rf = RandomForestRegressor(n_estimators=len(unpacked_model))
            else:
                rf = RandomForestClassifier(n_estimators=len(unpacked_model))
                rf.classes_ = np.array([0, 1], dtype=np.int64)
                rf.n_classes_ = 2
                
            rf.estimators_ = unpacked_model
            rf.n_features_in_ = self._cached_X_test.shape[1]
            rf.n_outputs_ = 1
            y_pred = rf.predict(self._cached_X_test)
        else:
            y_pred = unpacked_model.predict(self._cached_X_test)
            
        return pickle.dumps({
            "y_pred": y_pred,
            "y_true": self._cached_y_test,
            "n_samples": self.local_sample_count
        })

   
    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference

    def exposed_get_local_y_test(self) -> bytes:
        if self._cached_y_test is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self._cached_y_test)
    
    def exposed_get_local_sample_count(self) -> int:
        return self.local_sample_count