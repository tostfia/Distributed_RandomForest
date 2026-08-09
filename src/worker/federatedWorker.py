from multiprocessing.pool import Pool
import os
import pickle
from time import time
import time as time_module
from botocore.exceptions import ClientError
import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from rpyc.utils.classic import obtain
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
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
    
    tree_seed, max_depth, max_samples, bootstrap, tree_class, class_weight = args
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
        
    # 2. Prepariamo i parametri per l'inizializzazione dell'albero in modo dinamico
    kwargs = {"random_state": tree_seed}
    
    if max_depth is not None:
        kwargs["max_depth"] = max_depth
        
    # 3. CONTROLLO CRUCIALE: Aggiungiamo class_weight solo se l'albero è un classificatore
    if class_weight is not None and "Classifier" in tree_class.__name__:
        kwargs["class_weight"] = class_weight
        
    # Istanziazione dell'albero specifico richiesto con i parametri validati
    tree = tree_class(**kwargs)
        
    tree.fit(X_sampled, y_sampled)
    return tree

class FederatedWorker(BaseWorker):
    """Worker per la gestione dell'addestramento in modalità federata.

    Interpreta le stringhe sintetiche oppure scarica lo shard reale assegnato
    dall'Orchestratore salvandolo nella cache del disco rigido locale.
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
        
        # Cache dello stato interno
        self._cached_job_id = None
        self._cached_X_train = None
        self._cached_y_train = None
        self._cached_X_test = None
        self._cached_y_test = None
        self.local_sample_count = 0
        
        # Manteniamo aperto il file di lock come variabile d'istanza per impedire al Garbage Collector di distruggerlo
        self._index_lock_file = None

        # --- GESTIONE COERENTE DEL REGISTRO DEI WORKER (FAULT-TOLERANT) ---
        if self.environment == "aws":
            self.worker_index = 1
            for char in worker_name.split("-"):
                if char.isdigit():
                    self.worker_index = int(char)
                    break
            self.local_cache_dir = f"/tmp/{worker_name}_cache"
        else:
            # Siamo in ambiente "local" (Docker Compose Replicas)
            # Acquisiamo un indice atomico non-bloccante tramite fcntl
            num_workers = int(os.environ.get("NUM_WORKERS", 2))
            self.worker_index = self._claim_worker_index(num_workers)
            
            # Generiamo il nome uniforme corrispondente alle directory generate dallo splitter dell'orchestratore
            worker_id_uniforme = f"Worker-Locale-{self.worker_index:02d}"
            self.local_cache_dir = os.path.join("./workers_cache", worker_id_uniforme)
            
        os.makedirs(self.local_cache_dir, exist_ok=True)

        print(
            f"[{self.worker_name}] Inizializzato con successo. "
            f"Indice Worker: {self.worker_index} — Cache Dir: {self.local_cache_dir}"
        )

    def _claim_worker_index(self, num_workers: int) -> int:
        import fcntl
        
        lock_dir = os.environ.get("LOCAL_STORAGE_PATH", os.path.abspath("./.local_storage"))
        os.makedirs(lock_dir, exist_ok=True)

        # Scansione atomica degli indici disponibili da 1 a NUM_WORKERS
        for candidate in range(1, num_workers + 1):
            lock_path = os.path.join(lock_dir, f"worker_{candidate}.lock")
            
            try:
                # Apertura del file pointer associato alla sedia numerica
                f = open(lock_path, "w")
                
                # Tentativo di acquisizione del lock esclusivo NON BLOCCANTE
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Se la chiamata non genera BlockingIOError, la sedia è libera!
                f.write(f"Occupato da {self.worker_name} al timestamp {time_module.time()}\n")
                f.flush()
                
                # Preserviamo l'oggetto file aperto nell'istanza per mantenere attivo il lock a livello di kernel
                self._index_lock_file = f 
                
                print(f"[{self.worker_name}] [AUTO-INDEX] Conquistato con successo l'indice stazionario: {candidate}")
                return candidate

            except BlockingIOError:
                # Sedia occupata da un'altra replica concorrente, chiudiamo il descrittore locale e passiamo oltre
                f.close()
                continue

        # Estremo fallback protettivo
        print(f"[{self.worker_name}] [ATTENZIONE] Nessuna sedia libera trovata in .local_storage. Fallback forzato su indice 1.")
        return 1

    def is_regression(self) -> bool:
        return self.tree_type == "regressor"
    
    def _get_tree_class(self) -> type:
        return self.tree_class_reference
   
    def _load_data(self, dataset_tag: str):
        dao = DatasetDAOFactory.get_dao(self.environment)
        
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
        return self._load_data("real")
    
    def exposed_train_local_federated_forest(self, job_id: str, dataset_type: str, n_estimators_local: int, worker_index: int, hyperparameters: dict) -> list:
        hyperparameters = obtain(hyperparameters)
        max_depth = hyperparameters.get("max_depth")
        base_seed = int(hyperparameters.get("random_state", 123))

        self.tree_type = hyperparameters.get("tree_type", "classifier")
        self.target_column = "Target" if self.is_regression() else "Label"
        
        print(f"\n[{self.worker_name}] Ricevuto Task RPC Federato per Job {job_id[:8]}")

        if self._cached_job_id != job_id or self._cached_X_train is None:
            if dataset_type == "synthetic":
                self._load_synthetic_data(hyperparameters)
            else:
                self._load_and_preprocess_real_shard(worker_index, hyperparameters)
            self._cached_job_id = job_id

        tree_type = hyperparameters.get("tree_type", "classifier")
        if tree_type == "classifier":
            tree_class = DecisionTreeClassifier
        else:
            tree_class = DecisionTreeRegressor
        totale_core = os.cpu_count() or 1
        allocated_cores = max(1, totale_core - 1) if totale_core > 2 else totale_core
        
        class_weight = hyperparameters.get("class_weight", None)
        worker_tasks = []
        for i in range(n_estimators_local):
            seed = base_seed + i
            worker_tasks.append((seed, max_depth, self.max_samples, self.bootstrap, tree_class, class_weight))

        if n_estimators_local == 1:
            print(f"[{self.worker_name}] Ottimizzazione: 1 solo albero. Calcolo diretto senza Pool.")
            global _fed_child_X, _fed_child_y
            _fed_child_X = self._cached_X_train
            _fed_child_y = self._cached_y_train
            local_trees = [_train_single_fed_tree(worker_tasks[0])]
        else:
            pool_size = min(n_estimators_local, allocated_cores)
            print(f"[{self.worker_name}] Istanziazione Pool locale indipendente con {pool_size} processi...")
            
            with Pool(processes=pool_size, initializer=_init_fed_child_process, initargs=(self._cached_X_train, self._cached_y_train)) as pool:
                print(f"[{self.worker_name}] Mapping parallelo in corso su shard locale...")
                local_trees = pool.map(_train_single_fed_tree, worker_tasks)

        print(f"[+] [{self.worker_name}] Addestramento completato. Serializzazione in corso...")
        return pickle.dumps(local_trees)
    
    def _load_and_preprocess_real_shard(self, worker_index: int, hyperparameters: dict):
        train_filename = "train_shard.csv"
        test_filename = "test_shard.csv"
        local_train_path = os.path.join(self.local_cache_dir, train_filename)
        local_test_path = os.path.join(self.local_cache_dir, test_filename)

        hyperparameters = obtain(hyperparameters)

        if self.environment == "aws":
            bucket_name = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")
            s3_train_key = f"federated_shards/worker_{worker_index}/{train_filename}"
            s3_test_key = f"federated_shards/worker_{worker_index}/{test_filename}"
            
            s3_client = boto3.client("s3")
            try:
                s3_client.download_file(bucket_name, s3_train_key, local_train_path)
                s3_client.download_file(bucket_name, s3_test_key, local_test_path)
            except ClientError as e:
                raise IOError(f"[{self.worker_name}] Errore download shard da S3: {e}")

        if not os.path.exists(local_train_path):
            raise FileNotFoundError(f"Shard non trovato in {local_train_path}")
            
        df_train_raw = pd.read_csv(local_train_path, low_memory=False)
        df_test_raw = pd.read_csv(local_test_path, low_memory=False)

        preprocessor = CICIDSPreprocessor(target_column=self.target_column)
        
        df_train_bin = preprocessor.binarize_target(df_train_raw)
        df_test_bin = preprocessor.binarize_target(df_test_raw)
        
        df_train_clean = preprocessor.process(df_train_bin)
        df_test_clean = preprocessor.process(df_test_bin)

        selected_features = hyperparameters.get("feature_selezionate", None)
        
        if selected_features is not None:
            print(f"[{self.worker_name}] Applicazione spazio feature sincronizzato dal Master ({len(selected_features)} colonne).")
            features_to_keep = [col for col in selected_features if col != self.target_column]
            X_train_df = df_train_clean[features_to_keep]
            X_test_df = df_test_clean[features_to_keep]
        else:
            print(f"[{self.worker_name}] [ATTENZIONE] Nessuna lista feature passata. Uso il set completo.")
            X_train_df = df_train_clean.drop(columns=[self.target_column])
            X_test_df = df_test_clean.drop(columns=[self.target_column])

        y_train_df = df_train_clean[self.target_column]
        y_test_df = df_test_clean[self.target_column]

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
        hyperparameters = obtain(hyperparameters)
        seed = hyperparameters.get("random_state", 123)
        task = "regression" if self.is_regression() else "classification"
        target_column = "Target" if task == "regression" else "Label"
        n_samples = 166666
        loader = SyntheticDataLoader(task=task,n_samples=n_samples, random_seed=seed, target_column=target_column,output_dir=self.local_cache_dir)
        df = loader.load()

        X = df.drop(columns=[target_column]).to_numpy(dtype=np.float64)
        y = df[target_column].to_numpy()

        if task == "classification":
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y
            )
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=seed
            )
        
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
        job_id = payload.get("job_id", None)
        worker_index = payload.get("worker_index", None)
        hyperparameters = payload.get("hyperparameters", {})
        dataset_type = hyperparameters.get("dataset_type", "real")
        tree_type = hyperparameters.get("tree_type", "classifier")

        if self._cached_job_id != job_id or self._cached_X_test is None or self._cached_y_test is None:
            print(f"[{self.worker_name}] Rigenerazione cache di test tramite pipeline ufficiale...")
            if dataset_type == "synthetic":
                self._load_synthetic_data(hyperparameters)
            else:
                self._load_and_preprocess_real_shard(worker_index, hyperparameters)
        self._cached_job_id = job_id

        unpacked_model = pickle.loads(forest)

        if isinstance(unpacked_model, list):
            from sklearn.tree import DecisionTreeRegressor as DTR
            actual_is_regressor = isinstance(unpacked_model[0], DTR)

            if actual_is_regressor != (tree_type == "regressor"):
                print(
                    f"[{self.worker_name}] [WARN] Mismatch tree_type: payload='{tree_type}' "
                    f"ma alberi ricevuti={'Regressor' if actual_is_regressor else 'Classifier'}. "
                    f"Uso il tipo reale degli alberi."
                )

            if actual_is_regressor:
                rf = RandomForestRegressor(n_estimators=len(unpacked_model))
            else:
                rf = RandomForestClassifier(n_estimators=len(unpacked_model))
                global_classes = hyperparameters.get("global_classes", [0, 1])
                rf.classes_ = np.array(global_classes, dtype=np.int64)
                rf.n_classes_ = len(rf.classes_)
                local_unique = np.unique(self._cached_y_test)
                if len(local_unique) < len(rf.classes_):
                    print(f"[{self.worker_name}] [WARN] Il test-shard locale contiene solo le classi {local_unique.tolist()} "
                        f"su {rf.classes_.tolist()} attese. Possibile shard sbilanciato o indice worker duplicato.")

            rf.estimators_ = unpacked_model
            rf.n_features_in_ = self._cached_X_test.shape[1]
            rf.n_outputs_ = 1
            y_pred = rf.predict(self._cached_X_test)
        else:
            y_pred = unpacked_model.predict(self._cached_X_test)

        return pickle.dumps({
            "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
            "y_true": self._cached_y_test.tolist() if isinstance(self._cached_y_test, np.ndarray) else list(self._cached_y_test),
            "n_samples": len(self._cached_X_test)
        })
    

    def exposed_get_local_y_test(self) -> bytes:
        if self._cached_y_test is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self._cached_y_test)
    
    def exposed_get_local_sample_count(self) -> int:
        return self.local_sample_count