import json
import os
import pickle
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from multiprocessing import shared_memory

_fed_child_X = None
_fed_child_y = None
_shm_X_ref = None  # tiene vivo il blocco nel processo figlio
_shm_y_ref = None


RPC_SYNC_TIMEOUT_SECONDS = int(os.environ.get("RPC_SYNC_TIMEOUT_SECONDS", 1800))

def _init_fed_child_process(shm_name_X, shape_X, dtype_X, shm_name_y, shape_y, dtype_y):
    """Inizializza il processo figlio agganciandosi allo shared memory invece
    di ricevere una copia pickle-ata di X/y (che con 'spawn' verrebbe
    duplicata integralmente per OGNI processo, moltiplicando l'uso di RAM
    per pool_size)."""
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    global _fed_child_X, _fed_child_y, _shm_X_ref, _shm_y_ref
    _shm_X_ref = shared_memory.SharedMemory(name=shm_name_X)
    _fed_child_X = np.ndarray(shape_X, dtype=dtype_X, buffer=_shm_X_ref.buf)
    _shm_y_ref = shared_memory.SharedMemory(name=shm_name_y)
    _fed_child_y = np.ndarray(shape_y, dtype=dtype_y, buffer=_shm_y_ref.buf)

def _train_single_fed_tree(args):
    global _fed_child_X, _fed_child_y
    tree_seed, max_depth, max_samples, bootstrap, tree_class, class_weight, max_features, min_samples_split, criterion = args
    np.random.seed(tree_seed)

    n_samples = _fed_child_X.shape[0]

    if bootstrap:
        if max_samples is None:
            num_samples_to_draw = n_samples
        elif isinstance(max_samples, float):
            num_samples_to_draw = int(max_samples * n_samples)
        else:
            num_samples_to_draw = max_samples

        indices = np.random.choice(n_samples, size=num_samples_to_draw, replace=True)
        # Zero-copy: invece di X_sampled = _fed_child_X[indices] (~572MB copiati),
        # calcoliamo quante volte ogni riga originale è stata estratta e passiamo
        # questo peso a tree.fit(). Matematicamente equivalente al duplicare le
        # righe (stesso identico calcolo di impurità Gini/MSE che fa sklearn
        # internamente in _parallel_build_trees), ma senza mai allocare una
        # copia fisica dell'array.
        sample_weight = np.bincount(indices, minlength=n_samples).astype(np.float64)
        X_fit, y_fit = _fed_child_X, _fed_child_y
    else:
        sample_weight = None
        X_fit, y_fit = _fed_child_X, _fed_child_y

    kwargs = {"random_state": tree_seed, "max_features": max_features, "min_samples_split": min_samples_split}
    if max_depth is not None:
        kwargs["max_depth"] = max_depth
    if criterion is not None:
        kwargs["criterion"] = criterion
    if class_weight is not None and "Classifier" in tree_class.__name__:
        kwargs["class_weight"] = class_weight

    tree = tree_class(**kwargs)
    tree.fit(X_fit, y_fit, sample_weight=sample_weight)
    return tree

class FederatedWorker(BaseWorker):
    """Worker per la gestione dell'addestramento in modalità federata.

    In ambiente AWS, il worker possiede GIÀ i propri dati (shard reale +
    manifesti di feature selection) prima ancora di registrarsi come
    disponibile: li scarica una volta sola nel proprio __init__, da un bucket
    S3 seminato in precedenza da uno script di provisioning standalone
    (scripts/provision_federated_shards.py). Nessun download o generazione di
    dati avviene più reattivamente durante un job — questo simula un vero
    scenario federato, dove il nodo nasce già con il proprio dataset locale.

    In ambiente locale (single machine) resta invece il comportamento
    precedente: il vincolo tecnico della macchina unica rende necessario un
    passaggio intermedio gestito dall'Orchestratore.
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
        # (usato solo dal branch locale, vedi _claim_worker_index)
        self._index_lock_file = None

        # --- ASSEGNAZIONE DELL'INDICE WORKER (= "quale shard mi appartiene") ---
        if self.environment == "aws":
            # Binding FISSO, deciso a priori in fase di provisioning/deploy
            # (es. user-data / launch template dell'istanza EC2), non più
            # reclamato dinamicamente a runtime: in un ambiente federato reale
            # un nodo non "sorteggia" il proprio dataset, lo possiede già.
            worker_index_env = os.environ.get("WORKER_INDEX")
            if not worker_index_env:
                raise RuntimeError(
                    f"[{worker_name}] Variabile d'ambiente WORKER_INDEX non impostata. "
                    f"In ambiente AWS ogni istanza worker deve ricevere un indice di shard "
                    f"fisso (1..N), assegnato in fase di provisioning/deploy — non è più "
                    f"reclamato dinamicamente via ServiceRegistry."
                )
            try:
                self.worker_index = int(worker_index_env)
            except ValueError:
                raise RuntimeError(
                    f"[{worker_name}] WORKER_INDEX='{worker_index_env}' non è un intero valido."
                )

            self.worker_name = f"Worker-WIDX{self.worker_index}-{worker_name}"
            self.local_cache_dir = f"/tmp/{self.worker_name}_cache"
            os.makedirs(self.local_cache_dir, exist_ok=True)

            dataset_type_env = os.environ.get("DATASET_TYPE", "real").strip().lower()
            if dataset_type_env == "synthetic":
                print(f"[{self.worker_name}] [PROVISIONING] Dataset SINTETICO: nessuno shard reale da "
                      f"scaricare (verrà generato autonomamente al momento del job).")
            else:
                self._bootstrap_local_data_from_s3()
        else:
            # Siamo in ambiente "local" (Docker Compose Replicas)
            # Acquisiamo un indice atomico non-bloccante tramite fcntl
            num_workers = int(os.environ.get("NUM_WORKERS", 2))
            self.worker_index = self._claim_worker_index(num_workers)
            # Generiamo il nome uniforme corrispondente alle directory generate dallo splitter dell'orchestratore
            worker_id_uniforme = f"Worker-Locale-{self.worker_index:02d}"
            self.local_cache_dir = os.path.join("./workers_cache", worker_id_uniforme)
            os.makedirs(self.local_cache_dir, exist_ok=True)
            # Iniettiamo l'indice stabile appena conquistato nel NOME del worker
            # (marcatore WIDX), esattamente come fa il ramo AWS sopra. Senza questo,
            # il nome registrato presso l'orchestratore resterebbe quello generico
            # passato dal compose (Worker-Locale-federated-<hostname-hash>), privo
            # di indice: _infer_worker_index non potrebbe estrarlo e ripiegherebbe
            # sulla POSIZIONE nella lista dei worker disponibili — un indice
            # instabile tra orchestratori diversi, che romperebbe sia il binding
            # worker<->shard sia la ripresa da checkpoint dell'inferenza dopo un
            # failover. Con WIDX nel nome, l'indice è deterministico ovunque.
            self.worker_name = f"Worker-WIDX{self.worker_index}-{worker_name}"

        print(
            f"[{self.worker_name}] Inizializzato con successo. "
            f"Indice Worker: {self.worker_index} — Cache Dir: {self.local_cache_dir}"
        )

    def _bootstrap_local_data_from_s3(self):
        """
        Scarica in modo sincrono, PRIMA che il worker si registri come
        disponibile, tutto ciò che gli serve per essere autonomo: il proprio
        shard del dataset reale e i manifesti di feature selection
        (config_real.json / config_synthetic.json), se presenti.

        Questi artefatti sono generati offline da uno script di provisioning
        dedicato (scripts/provision_federated_shards.py), MAI durante un job
        di training.
        """
        bucket_name = os.environ.get(
            "DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an"
        )
        s3_client = boto3.client("s3")

        print(f"[{self.worker_name}] [PROVISIONING] Download shard e manifesti da S3 (bucket: {bucket_name})...")

        # 1. Shard del dataset reale: OBBLIGATORIO. Se manca, il worker non
        #    può considerarsi operativo — meglio fallire subito ed
        #    esplicitamente che scoprirlo al primo job.
        shard_keys = {
            "train_shard.csv": f"federated_shards/worker_{self.worker_index}/train_shard.csv",
            "test_shard.csv": f"federated_shards/worker_{self.worker_index}/test_shard.csv",
        }
        for local_filename, s3_key in shard_keys.items():
            local_path = os.path.join(self.local_cache_dir, local_filename)
            try:
                s3_client.download_file(bucket_name, s3_key, local_path)
                print(f"[{self.worker_name}] [PROVISIONING] Scaricato {s3_key} -> {local_path}")
            except ClientError as e:
                raise IOError(
                    f"[{self.worker_name}] Impossibile scaricare lo shard '{s3_key}' dal bucket "
                    f"'{bucket_name}'. Hai eseguito lo script di provisioning "
                    f"(scripts/provision_federated_shards.py) prima di avviare i worker? Dettaglio: {e}"
                )

        # 2. Manifesti di feature selection: best-effort, possono non esistere
        #    entrambi (es. se la baseline è stata eseguita solo sul reale o
        #    solo sul sintetico). Il worker li terrà entrambi in cache e
        #    sceglierà quello giusto al momento del job, in base al
        #    dataset_type richiesto — senza bisogno che il master glieli
        #    inietti via RPC.
        for dataset_type in ("real", "synthetic"):
            config_filename = f"config_{dataset_type}.json"
            s3_key = f"federated_config/{config_filename}"
            local_path = os.path.join(self.local_cache_dir, config_filename)
            try:
                s3_client.download_file(bucket_name, s3_key, local_path)
                print(f"[{self.worker_name}] [PROVISIONING] Scaricato manifesto '{config_filename}'.")
            except ClientError:
                print(
                    f"[{self.worker_name}] [PROVISIONING] Manifesto '{config_filename}' non trovato su S3 "
                    f"(ok se non esegui job con dataset_type='{dataset_type}')."
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


    
    def exposed_train_local_federated_forest(self, job_id: str, dataset_type: str, n_estimators_local: int, worker_index: int, hyperparameters: dict) -> list:
        hyperparameters = obtain(hyperparameters)
        max_depth = hyperparameters.get("max_depth")
        base_seed = int(hyperparameters.get("random_state", 123))
        max_samples = hyperparameters.get("max_samples", self.max_samples) 

        self.tree_type = hyperparameters.get("tree_type", "classifier")
    
        # Fallback ai default "corretti" di RandomForest{Classifier,Regressor}
        # se il manifesto non specifica esplicitamente max_features.
        max_features = hyperparameters.get(
            "max_features", "sqrt" if not self.is_regression() else (1 / 3)
        )
        min_samples_split = hyperparameters.get("min_samples_split", 2)
        criterion = hyperparameters.get("criterion", None)
        self.target_column = "Target" if self.is_regression() else "Label"
        
        print(f"\n[{self.worker_name}] Ricevuto Task RPC Federato per Job {job_id[:8]}")

        if self._cached_job_id != job_id or self._cached_X_train is None:
            if dataset_type == "synthetic":
                self._load_synthetic_data(hyperparameters)
            else:
                self._load_and_preprocess_real_shard(worker_index, hyperparameters, dataset_type=dataset_type)
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
            worker_tasks.append((seed, max_depth, max_samples, self.bootstrap, tree_class, class_weight, max_features,
                                min_samples_split, criterion))

        if n_estimators_local == 1:
            print(f"[{self.worker_name}] Ottimizzazione: 1 solo albero. Calcolo diretto senza Pool.")
            global _fed_child_X, _fed_child_y
            _fed_child_X = self._cached_X_train
            _fed_child_y = self._cached_y_train
            local_trees = [_train_single_fed_tree(worker_tasks[0])]
        else:
            pool_size = min(n_estimators_local, allocated_cores)
            print(f"[{self.worker_name}] Istanziazione ThreadPool locale con {pool_size} thread "
                f"(memoria condivisa nativa, nessuna copia/serializzazione tra thread)...")

            # I thread condividono già la memoria del processo: niente shared_memory,
            # niente initializer, niente re-import per processo (spawn). _fed_child_X/y
            # restano semplici variabili globali del modulo, assegnate una sola volta qui.
            
            _fed_child_X = self._cached_X_train
            _fed_child_y = self._cached_y_train

            BATCH_SIZE = max(1, min(pool_size * 4, n_estimators_local))
            local_trees = []
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                for batch_start in range(0, len(worker_tasks), BATCH_SIZE):
                    batch = worker_tasks[batch_start: batch_start + BATCH_SIZE]
                    print(f"[{self.worker_name}] Batch alberi {batch_start}-{batch_start+len(batch)} "
                        f"di {n_estimators_local}...")
                    futures = [executor.submit(_train_single_fed_tree, task) for task in batch]
                    try:
                        batch_trees = [f.result(timeout=RPC_SYNC_TIMEOUT_SECONDS) for f in futures]
                    except TimeoutError:
                        raise RuntimeError(
                            f"[{self.worker_name}] Timeout ({RPC_SYNC_TIMEOUT_SECONDS}s) durante il "
                            f"training parallelo (ThreadPool)."
                        )
                    local_trees.extend(batch_trees)
                    del batch_trees

        print(f"[+] [{self.worker_name}] Addestramento completato. Salvataggio su storage condiviso...")
        serialized_task = pickle.dumps(local_trees)
        # Riusiamo lo stesso schema di storage del path centralizzato (vedi
        # BaseWorker._save_task_to_shared_storage / _get_task_storage_paths,
        # ereditati da questa classe): quelle funzioni derivano il job_id da un
        # 'source_info' del tipo 'shared_train_<job_id>.csv'. Il federato non usa
        # un path di dataset per il training locale (ogni worker ha già il
        # proprio shard in cache), quindi costruiamo una stringa sintetica solo
        # per ottenere la STESSA chiave che l'Orchestratore userà in lettura
        # (vedi federated.py, worker_thread_consumer).
        synthetic_source_info = f"shared_train_{job_id}.csv"
        try:
            self._save_task_to_shared_storage(synthetic_source_info, base_seed, n_estimators_local, serialized_task)
        except Exception as e:
            # Il task NON deve risultare "completato con successo" se non è stato
            # persistito nello storage condiviso: rilanciamo l'eccezione così RPyC
            # la propaga all'Orchestratore, che potrà marcare il task come fallito
            # e decidere se ritentarlo, invece di credere erroneamente che sia
            # andato tutto bene.
            print(f"[!] [{self.worker_name}] ERRORE CRITICO: gli alberi sono stati calcolati "
                  f"ma il salvataggio nello storage condiviso è fallito. Il task viene "
                  f"segnalato come fallito all'Orchestratore. Dettaglio: {e}")
            raise
        print(f"[+] [{self.worker_name}] Task salvato nello storage condiviso. Invio ack "
              f"(niente più blob via RPC).")
        # Non restituiamo più gli alberi per intero via RPyC: l'Orchestratore li
        # rilegge dallo storage condiviso con lo stesso 'source_info' sintetico
        # (vedi federated.py). Stesso fix già applicato al path centralizzato per
        # evitare l'hang su payload grandi come valore di ritorno RPC sincrono.
        return {"ack": True, "num_trees": n_estimators_local}

    def _resolve_selected_features(self, dataset_type: str, hyperparameters: dict):
        """
        Determina quale spazio di feature usare per il preprocessing dello shard.

        - In AWS: il worker le risolve in AUTONOMIA leggendo il manifesto
          già scaricato in cache al boot (config_{dataset_type}.json). Non
          dipende più da ciò che il master inietta via RPC.
        - In locale: comportamento invariato, la lista arriva ancora nel
          payload hyperparameters (iniettata dal master via select_from_config).
        """
        if self.environment == "aws":
            config_path = os.path.join(self.local_cache_dir, f"config_{dataset_type}.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    feature_selezionate = config_data.get("feature_selezionate")
                    if feature_selezionate:
                        print(
                            f"[{self.worker_name}] Feature space risolto localmente da "
                            f"'{config_path}' ({len(feature_selezionate)} colonne)."
                        )
                        return feature_selezionate
                except Exception as e:
                    print(f"[{self.worker_name}] [ATTENZIONE] Lettura manifesto locale fallita: {e}")
            print(
                f"[{self.worker_name}] [ATTENZIONE] Nessun manifesto locale trovato per "
                f"dataset_type='{dataset_type}'. Uso il set di feature completo."
            )
            return None

        return hyperparameters.get("feature_selezionate", None)

    def _load_and_preprocess_real_shard(self, worker_index: int, hyperparameters: dict, dataset_type: str = "real"):
        train_filename = "train_shard.csv"
        test_filename = "test_shard.csv"
        local_train_path = os.path.join(self.local_cache_dir, train_filename)
        local_test_path = os.path.join(self.local_cache_dir, test_filename)

        hyperparameters = obtain(hyperparameters)

        # In AWS lo shard è già stato scaricato in fase di provisioning/boot
        # (_bootstrap_local_data_from_s3, chiamato da __init__): qui leggiamo
        # solo dal disco locale, nessuna chiamata di rete durante il job.
        if not os.path.exists(local_train_path):
            hint = (
                " Il provisioning AWS non è stato eseguito o è fallito: lancia "
                "'python -m scripts.provision_federated_shards' prima di avviare i worker."
                if self.environment == "aws"
                else ""
            )
            raise FileNotFoundError(f"Shard non trovato in {local_train_path}.{hint}")
            
        df_train_raw = pd.read_csv(local_train_path, low_memory=False)
        df_test_raw = pd.read_csv(local_test_path, low_memory=False)

        preprocessor = CICIDSPreprocessor(target_column=self.target_column)
        
        df_train_bin = preprocessor.binarize_target(df_train_raw)
        df_test_bin = preprocessor.binarize_target(df_test_raw)
        
        df_train_clean = preprocessor.process(df_train_bin)
        df_test_clean = preprocessor.process(df_test_bin)

        selected_features = self._resolve_selected_features(dataset_type, hyperparameters)
        
        if selected_features is not None:
            print(f"[{self.worker_name}] Applicazione spazio feature sincronizzato ({len(selected_features)} colonne).")
            features_to_keep = [col for col in selected_features if col != self.target_column]
            X_train_df = df_train_clean[features_to_keep]
            X_test_df = df_test_clean[features_to_keep]
        else:
            print(f"[{self.worker_name}] [ATTENZIONE] Nessuna lista feature disponibile. Uso il set completo.")
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
        seed = hyperparameters.get("dataset_random_state", hyperparameters.get("random_state", 123))
        task = "regression" if self.is_regression() else "classification"
        target_column = "Target" if task == "regression" else "Label"
        n_samples = hyperparameters.get("n_samples", 166666)
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
        # 'model_path' (nuovo, preferito): il modello globale è già su storage
        # condiviso (l'Orchestratore lo ha appena caricato da lì per ottenere
        # 'all_trees'), quindi lo riscarichiamo e deserializziamo da soli
        # invece di riceverlo per intero via RPC. Prima veniva ripetuto questo
        # trasferimento (fino a 1+ GB) UNA VOLTA PER OGNI WORKER — peggio
        # ancora del path centralizzato, dove almeno viene diviso in chunk.
        # 'forest' (retrocompatibilità): byte già serializzati, usati se
        # 'model_path' non è presente.
        model_path = payload.get("model_path")
        forest = payload.get("forest")
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
                self._load_and_preprocess_real_shard(worker_index, hyperparameters, dataset_type=dataset_type)
        self._cached_job_id = job_id

        if model_path is not None:
            from src.dataset.checkpoint_dao import CheckpointDAOFactory
            print(f"[{self.worker_name}] Caricamento del modello globale da storage condiviso: {model_path}")
            checkpoint_dao = CheckpointDAOFactory.get_dao(self.environment)
            loaded_model = checkpoint_dao.load(model_path)
            # Normalizziamo sempre a LISTA di alberi (coerente col resto del
            # metodo, che si aspetta 'unpacked_model' come lista): il file
            # salvato è l'oggetto RandomForest{Classifier,Regressor} completo.
            unpacked_model = loaded_model.estimators_ if hasattr(loaded_model, "estimators_") else loaded_model
        elif forest is not None:
            unpacked_model = pickle.loads(forest)
        else:
            raise ValueError(
                f"[{self.worker_name}] Payload di inferenza privo sia di 'model_path' sia di 'forest'."
            )

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
                rf = RandomForestRegressor(n_estimators=len(unpacked_model), n_jobs=-1)
            else:
                rf = RandomForestClassifier(n_estimators=len(unpacked_model),n_jobs=-1)
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

            y_probs = None
            if not actual_is_regressor and len(rf.classes_) == 2:
                proba_matrix = rf.predict_proba(self._cached_X_test)
                positive_label = rf.classes_[-1]  # convenzione: classe con etichetta maggiore = positiva (es. 1 in 0/1)
                positive_idx = int(np.where(rf.classes_ == positive_label)[0][0])
                if proba_matrix.shape[1] == len(rf.classes_):
                    y_probs = proba_matrix[:, positive_idx]
                else:
                    print(f"[{self.worker_name}] [WARN] predict_proba ha restituito "
                          f"{proba_matrix.shape[1]} colonne, attese {len(rf.classes_)}: "
                          f"AUC non calcolabile su questo worker.")
        else:
            y_pred = unpacked_model.predict(self._cached_X_test)
            y_probs = None
            if tree_type == "classifier" and hasattr(unpacked_model, "predict_proba"):
                try:
                    classes = getattr(unpacked_model, "classes_", None)
                    if classes is not None and len(classes) == 2:
                        proba_matrix = unpacked_model.predict_proba(self._cached_X_test)
                        positive_idx = int(np.where(classes == classes[-1])[0][0])
                        y_probs = proba_matrix[:, positive_idx]
                except Exception as e:
                    print(f"[{self.worker_name}] [WARN] predict_proba non disponibile su questo modello: {e}")

        response = {
            "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
            "y_true": self._cached_y_test.tolist() if isinstance(self._cached_y_test, np.ndarray) else list(self._cached_y_test),
            "n_samples": len(self._cached_X_test)
        }
        if y_probs is not None:
            response["y_probs"] = y_probs.tolist() if isinstance(y_probs, np.ndarray) else list(y_probs)

        return pickle.dumps(response)
    

    def exposed_get_local_y_test(self) -> bytes:
        if self._cached_y_test is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self._cached_y_test)
    
    def exposed_get_local_sample_count(self) -> int:
        return self.local_sample_count

    def exposed_get_local_shard_size(self) -> int:
        """
        Numero di righe nel training-shard locale, letto direttamente dal CSV
        su disco SENZA preprocessing/feature-selection. Serve all'Orchestratore
        PRIMA di avviare un round di training, per allocare il budget di alberi
        in proporzione alla quantità di dati posseduti da ciascun worker invece
        che in parti uguali — indispensabile con partizionamento non-IID
        (dirichlet/by_day), dove gli shard possono avere dimensioni molto
        diverse tra loro; con partizionamento IID il risultato è comunque
        praticamente identico alla vecchia ripartizione equa.

        Approssimazione voluta: il conteggio grezzo (righe prima della pulizia
        NaN/inf) è una stima sufficiente ai fini di un'allocazione di risorse
        (la pulizia tipicamente rimuove una frazione minima delle righe) ed
        evita di dover rieseguire l'intero preprocessing solo per contare i
        campioni, prima ancora di sapere se il worker verrà davvero usato in
        questo round.

        Per dataset_type='synthetic' il file non esiste ancora a questo punto
        (i dati sintetici vengono generati pigramente al primo training, non
        al boot): ritorniamo 0, e l'Orchestratore ricade sulla ripartizione
        equa storica quando NESSUN worker restituisce una dimensione nota.
        """
        train_path = os.path.join(self.local_cache_dir, "train_shard.csv")
        if not os.path.exists(train_path):
            print(f"[{self.worker_name}] [INFO] Shard locale non ancora presente in {train_path} "
                  f"(probabile dataset sintetico): ritorno dimensione 0.")
            return 0
        try:
            with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
                n_lines = sum(1 for _ in f)
            return max(0, n_lines - 1)  # -1 per l'header, mai negativo
        except Exception as e:
            print(f"[{self.worker_name}] [WARN] Errore nel conteggio righe di {train_path}: {e}. Ritorno 0.")
            return 0