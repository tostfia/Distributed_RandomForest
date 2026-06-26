from multiprocessing.pool import Pool
import os
import pickle
from botocore.exceptions import ClientError
import boto3
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from rpyc.utils.classic import obtain
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.worker.BaseWorker import BaseWorker
from src.shared.factory import DatasetDAOFactory 

_fed_child_X = None
_fed_child_y = None

def _init_fed_child_process(X, y):
    """Inizializza il processo figlio del pool federato isolando i thread della CPU."""
    import os
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    global _fed_child_X, _fed_child_y
    _fed_child_X = X
    _fed_child_y = y

def _train_single_fed_tree(args):
    """Esegue l'addestramento attingendo dallo shard in memoria globale del processo figlio."""
    
    global _fed_child_X, _fed_child_y
    
    tree_seed, max_depth, max_samples, bootstrap, tree_class = args
    np.random.seed(tree_seed)
    
    n_samples = _fed_child_X.shape[0]
    
    # Gestione del bootstrap e del campionamento locale
    if bootstrap:
        if max_samples is None:
            num_samples_to_draw = n_samples
        elif isinstance(max_samples, float):
            num_samples_to_draw = int(max_samples * n_samples)
        else:
            num_samples_to_draw = max_samples
            
        indices = np.random.choice(n_samples, size=num_samples_to_draw, replace=True)
        X_sampled = _fed_child_X[indices]
        y_sampled = _fed_child_y[indices]
    else:
        X_sampled = _fed_child_X
        y_sampled = _fed_child_y
        
    # Istanziazione dell'albero specifico richiesto
    if max_depth is not None:
        tree = tree_class(random_state=tree_seed, max_depth=max_depth)
    else:
        tree = tree_class(random_state=tree_seed)
        
    tree.fit(X_sampled, y_sampled)
    return tree

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
    
    def exposed_train_local_federated_forest(self,job_id: str, dataset_type: str, n_estimators_local: int, worker_index: int, hyperparameters: dict) -> list:
                                              
        """Metodo esposto tramite RPC richiesto dall'Orchestratore per avviare l'addestramento.
        Chiama il caricamento dinamico dei dati e restituisce gli alberi locali in formato binario.
        """
        """Metodo RPC esposto all'Orchestratore per l'addestramento locale."""
        
        hyperparameters = obtain(hyperparameters)
        max_depth = hyperparameters.get("max_depth")
        base_seed = int(hyperparameters.get("random_state", 123))
        
        print(f"\n[{self.worker_name}] Ricevuto Task RPC Federato per Job {job_id[:8]}")

        # Caricamento e Preprocessing dei dati (eseguito solo se cambia il Job ID o la cache è vuota)
        if self._cached_job_id != job_id or self._cached_X_train is None:
            if dataset_type == "synthetic":
                self._load_synthetic_data(hyperparameters)
            else:
                # Passiamo l'intero payload per estrarre la lista globale delle feature sincronizzate
                self._load_and_preprocess_real_shard(worker_index, hyperparameters)
            self._cached_job_id = job_id

        
        tree_class = self._get_tree_class()
        totale_core = os.cpu_count() or 1
        allocated_cores = max(1, totale_core - 1) if totale_core > 2 else totale_core

        worker_tasks = []
        for i in range(n_estimators_local):
            seed = base_seed + i
            worker_tasks.append((seed, max_depth, self.max_samples, self.bootstrap, tree_class))

        # Addestramento parallelo locale
        # 5. Ottimizzazione anti-crash: esecuzione diretta se richiesto un solo albero
        if n_estimators_local == 1:
            print(f"[{self.worker_name}] Ottimizzazione: 1 solo albero. Calcolo diretto senza Pool.")
            global _fed_child_X, _fed_child_y
            _fed_child_X = self._cached_X_train
            _fed_child_y = self._cached_y_train
            local_trees = [_train_single_fed_tree(worker_tasks[0])]
        else:
            # Creazione di un pool locale, temporaneo e isolato per questo round federato
            pool_size = min(n_estimators_local, allocated_cores)
            print(f"[{self.worker_name}] Istanziazione Pool locale indipendente con {pool_size} processi...")
            
            with Pool(processes=pool_size, initializer=_init_fed_child_process, initargs=(self._cached_X_train, self._cached_y_train)) as pool:
                print(f"[{self.worker_name}] Mapping parallelo in corso su shard locale...")
                local_trees = pool.map(_train_single_fed_tree, worker_tasks)

        print(f"[+] [{self.worker_name}] Addestramento completato. Serializzazione in corso...")
        
        # 6. Ritorno diretto all'orchestratore
        return pickle.dumps(local_trees)
    
    def _load_and_preprocess_real_shard(self, worker_index: int, hyperparameters: dict):
        """Scarica e processa lo shard reale garantendo l'allineamento delle feature."""
        train_filename = "train_shard.csv"
        test_filename = "test_shard.csv"
        local_train_path = os.path.join(self.local_cache_dir, train_filename)
        local_test_path = os.path.join(self.local_cache_dir, test_filename)

        hyperparameters = obtain(hyperparameters)

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
        
        selected_features = hyperparameters.get("feature_selezionate", None)
        
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
        hyperparameters = obtain(hyperparameters)
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

    def exposed_predict_subset_forest(self, payload: dict) -> bytes:
        payload = pickle.loads(payload)
        forest = payload["forest"]

        if self._cached_X_test is None or self._cached_y_test is None:
              # Forza il caricamento dello shard reale se non presente
            print(f"[{self.worker_name}] Cache vuota. Ricarico lo shard locale in memoria...")
            iperparametri = payload.get("hyperparameters", {})
            self._load_and_preprocess_real_shard(self.worker_index, iperparametri)

   
        unpacked_model = pickle.loads(forest)
        
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
            "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
            "y_true": self._cached_y_test.tolist() if isinstance(self._cached_y_test, np.ndarray) else list(self._cached_y_test),
            "n_samples": self.local_sample_count
        })

   

    def exposed_get_local_y_test(self) -> bytes:
        if self._cached_y_test is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self._cached_y_test)
    
    def exposed_get_local_sample_count(self) -> int:
        return self.local_sample_count