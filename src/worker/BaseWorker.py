from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import os
import signal
import socket
import numpy as np
from rpyc import Service, ThreadedServer
import threading
import time
import pickle 
import boto3
import json
from botocore.exceptions import ClientError

from src.shared.config import SystemConfig
from src.shared.binding.serviceregistry import ServiceRegistry

_child_X  = None
_child_y  = None

def _init_child_process(X, y):
    """Inizializza il processo figlio salvando i dati in memoria e isolando i core."""
    
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    global _child_X, _child_y
    _child_X = X
    _child_y = y

# Addestramento di un singolo albero
def _train_single_tree_processor(args):
    """Esegue l'addestramento prelevando X e y dalla memoria globale del processo."""
    global _child_X, _child_y
    tree_seed, max_depth, max_samples, bootstrap, tree_class, max_features = args
    np.random.seed(tree_seed)
    
    # Preleviamo la shape direttamente dalla memoria condivisa del processo figlio
    n_samples = _child_X.shape[0]

    if bootstrap:
        size = int(max_samples * n_samples) if max_samples else n_samples
        indices = np.random.choice(n_samples, size=size, replace=True)
        X_train, y_train = _child_X[indices], _child_y[indices]
    else: 
        X_train, y_train = _child_X, _child_y
   
    # max_features attiva il sottocampionamento casuale delle feature ad ogni
    # split: è ciò che decorrela gli alberi tra loro (Breiman, 2001) e
    # distingue un vero Random Forest da un semplice bagging di alberi.
    # random_state passato esplicitamente invece di affidarsi solo al seed
    # globale np.random.seed sopra, per coerenza col path federato.
    tree = tree_class(
        splitter="best",
        max_depth=max_depth,
        max_features=max_features,
        random_state=tree_seed,
    )
    tree.fit(X_train, y_train)
    return tree

class BaseWorker(Service, ABC): 
    def __init__(
        self, 
        worker_name: str, 
        queue_name: str, 
        tree_class_reference, 
        url_dataset: str = None, 
        max_samples=None, 
        bootstrap: bool = True
    ):
        super().__init__() 
        # 1. Carichiamo la configurazione centralizzata dal file .env
        self.cfg = SystemConfig()
        self.environment = self.cfg.env
        
        self.worker_name = worker_name
        self.queue_name = queue_name
        self.url_dataset = url_dataset
        self.tree_class_reference = tree_class_reference
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self._stop_heartbeat = None

        self._cached_pool  = None
        self._cached_pool_source = None
        self._cached_X_test_bytes = None
        self._cached_X_eval = None


    @abstractmethod
    def is_regression(self):
        pass

    def release_index_claim(self):
        """Hook per sottoclassi che gestiscono claim di risorse condivise
        (es. FederatedWorker con l'indice shard su AWS). Implementazione di
        base: nessuna azione (usata da CentralizedWorker)."""
        pass

    def _get_my_private_ip(self) -> str:
        """Determina l'IP corretto per il binding di rete in base all'ambiente.
            Funziona sia in locale (con/senza Docker) sia su AWS.
        """
        # Se siamo dentro Docker (Compose imposta solitamente variabili o hostname specifici)
        # o se siamo su AWS, dobbiamo ascoltare su tutte le interfacce (0.0.0.0)
        if self.environment == "aws" or os.environ.get("RUNNING_IN_DOCKER", "false") == "true":
            return "0.0.0.0"
        
        # Locale puro senza Docker
        return "127.0.0.1"

    def start_server(self, port: int, explicit_host: str = None):
        print(f"\n[{self.worker_name}] Inizializzazione Server RPC in ambiente {self.environment.upper()}...")

        def _handle_sigterm(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, _handle_sigterm)

        advertise_host = os.environ.get("RPC_ADVERTISE_HOST", None)
        is_docker = os.environ.get("RUNNING_IN_DOCKER", "false") == "true"

        # Gestione degli host dinamica
        if self.environment == "aws" or is_docker:
            host_to_bind = "0.0.0.0"  # Permette a RPyC di accettare connessioni esterne/da altri container
            # Se siamo in Docker e non c'è un advertise_host esplicito, usiamo il socket hostname (il nome del container)
            host_to_register = advertise_host if advertise_host else socket.gethostname()
        else:
            # Locale nativo senza Docker
            host_to_bind = explicit_host if explicit_host else "127.0.0.1"
            host_to_register = host_to_bind

        print(f"[{self.worker_name}] Binding su: {host_to_bind}, Registrazione su Registry come: {host_to_register}:{port}")
        
        # Registrazione del Worker sul Service Registry
        ServiceRegistry.register_worker(worker_name=self.worker_name, host=host_to_register, port=port)

        self._stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, args=(self._stop_heartbeat, 10), daemon=True)
        heartbeat_thread.start()
        print(f"[+] [{self.worker_name}] Thread di Heartbeat avviato con successo.")

        protocol_config = {
            'allow_public_attr': True,
            'allow_pickle': True,
            'sync_request_timeout': 600, 
            'keepalive': True
        }

        server = ThreadedServer(self, hostname=host_to_bind, port=port, protocol_config=protocol_config)

        print("\n ==============================================")
        print(f"  SERVER WORKER IN ASCOLTO: {self.worker_name.upper()}")
        print(f"  Indirizzo di ascolto:    {host_to_bind}:{port}")
        print(f"  Ambiente attivo:          {self.environment.upper()}")
        print(" ==============================================\n")

        try:
            server.start()
        except KeyboardInterrupt:
            print(f"\n[-][{self.worker_name} Interruzione manuale rilevata]")
        except Exception as e:
            print(f"\n[!] [{self.worker_name}] Errore durante l'esecuzione del server: {str(e)}")
        finally:
            print(f"\n[+] [{self.worker_name}] Arresto del server in corso...")
            self._stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)

            self.release_index_claim()
            
            if self._cached_pool is not None:
                self._cached_pool.close()
                self._cached_pool.join()
                print(f"[+] [{self.worker_name}] Pool di processi chiuso correttamente.")
            try:
                ServiceRegistry.deregister_worker(self.worker_name)
                print(f"[+] [{self.worker_name}] Server arrestato e worker rimosso dal Service Registry.") 
            except Exception as e:
                print(f"[!] [{self.worker_name}] Errore durante la deregistrazione: {str(e)}")

    def _heartbeat_loop(self, stop_event: threading.Event, interval: int = 10):
        while not stop_event.is_set():
            try:
                ServiceRegistry.update_worker_heartbeat(self.worker_name)
            except Exception as e:
                print(f"[!] [{self.worker_name}] Errore durante l'invio dell'heartbeat: {str(e)}")
            for _ in range(interval):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    def on_connect(self, conn):
        peer_info = "Orchestratore"
        if hasattr(conn, '_config') and 'peer' in conn._config:
            peer_info = conn._config['peer']
        print(f"[+] Connessione stabilita con successo: {peer_info}")

    def on_disconnect(self, conn):
        print(f"[-] Connessione chiusa dall'Orchestratore.")

    @abstractmethod
    def _load_data(self, source_info):
        pass

    @abstractmethod
    def _get_tree_class(self):
        pass

    def exposed_train_subset_forest(self, source_info, num_trees, base_seed, max_depth=None, tree_type=None, max_features=None):
        print("\n=============================================================")
        print(f" [WORKER RPC] Richiesta elaborazione foresta parziale | Alberi: {num_trees}")
        print("=============================================================\n")
        if tree_type is not None:
            self.tree_type = tree_type
        # Se l'Orchestratore non specifica max_features (manifesti vecchi/non
        # aggiornati), ricadiamo sui default "corretti" di un vero Random
        # Forest invece che su None (= tutte le feature ad ogni split).
        if max_features is None:
            max_features = "sqrt" if not self.is_regression() else (1 / 3)
        cached_task_bytes = self._load_task_from_shared_storage(source_info, base_seed, num_trees)
        if cached_task_bytes is not None:
            print(f"[{self.worker_name}] [SHORT-CIRCUIT] Task già pronto nello storage. Restituisco i byte.")
            return cached_task_bytes
        # 1. Recupero dati e classe dell'albero dalle classi figlie
        X, y = self._load_data(source_info)
        tree_class = self._get_tree_class()

        # 2. CALCOLO DINAMICO DEI CORE
        # NOTA: la divisione dei core tra worker deve avvenire ogni volta che più worker
        # condividono la STESSA macchina fisica, indipendentemente dal fatto che il flag
        # .env sia "aws" o "local" (quel flag indica solo quale storage/coda usare,
        # non se i worker sono co-locati). Rileviamo la co-locazione interrogando
        # sempre il ServiceRegistry; se la query fallisce o c'è un solo worker,
        # ricadiamo sulla regola N-1 standard.
        totale_core_macchina = os.cpu_count() or 1

        try:
            workers_attivi = ServiceRegistry.get_available_workers(self.environment)
            num_workers = max(1, len(workers_attivi))

            if num_workers > 1:
                # Più worker rilevati: dividiamo i core disponibili tra tutti quelli
                # effettivamente attivi, per evitare sovra-allocazione quando sono
                # co-locati sulla stessa macchina fisica.
                core_disponibili_rete = max(1, totale_core_macchina - 1)
                allocated_cores = max(1, int(core_disponibili_rete / num_workers))
                print(f"[{self.worker_name}] [LOG] Rilevati {num_workers} worker attivi (ambiente: {self.environment}).")
                print(f"[{self.worker_name}] [LOG] Allocazione dinamica: {allocated_cores} processi per questo pool.")
            else:
                # Un solo worker rilevato: presumibilmente ha la macchina tutta per sé.
                allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina
        except Exception as e:
            print(f"[!] Errore lettura ServiceRegistry, fallback su N-1: {e}")
            allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina

        # 3. Ottimizzazione anti-crash per il multiprocessing in Docker
        if num_trees == 1:
            print("[WORKER] Ottimizzazione: 1 solo albero richiesto. Esecuzione diretta senza Pool.")
            direct_task  = (base_seed, max_depth, self.max_samples, self.bootstrap, tree_class, max_features)

            global _child_X, _child_y
            _old_child_X, _old_child_y = _child_X, _child_y
            _child_X, _child_y = X, y
            try:
                local_trees = [_train_single_tree_processor(direct_task)]
            finally:
                _child_X, _child_y = _old_child_X, _old_child_y
        else:
            if self._cached_pool_source != source_info or self._cached_pool is None:
                if self._cached_pool is not None:
                    self._cached_pool.close()
                    self._cached_pool.join()
                    print(f"[+] [{self.worker_name}] Pool di processi chiuso correttamente.")
                
                pool_size = min(num_trees, allocated_cores)
                print(f"[WORKER] Creazione nuovo Pool di processi calibrato a {pool_size} processi...")
                self._cached_pool = Pool(processes=pool_size, initializer=_init_child_process, initargs=(X, y))
                self._cached_pool_source = source_info
                
            print(f"[WORKER] Pool di processi pronto. Avvio addestramento di {num_trees} alberi...")
            worker_tasks = []
            for i in range(num_trees):
                seed = base_seed + i
                worker_tasks.append((seed, max_depth, self.max_samples, self.bootstrap, tree_class, max_features))
            local_trees = self._cached_pool.map(_train_single_tree_processor, worker_tasks)

        print(f"[+] Calcolo di {num_trees} alberi completato. Invio in corso via pickle...")
        serialized_task = pickle.dumps(local_trees)
        try:
            self._save_task_to_shared_storage(source_info, base_seed, num_trees, serialized_task)
        except Exception as e:
            # Il task NON deve risultare "completato con successo" se non è stato
            # persistito nello storage condiviso: rilanciamo l'eccezione così RPyC
            # la propaga all'Orchestratore, che potrà marcare il task come fallito
            # e decidere se ritentarlo, invece di credere erroneamente che sia andato
            # tutto bene (comportamento precedente, silenziosamente errato).
            print(f"[!] [{self.worker_name}] ERRORE CRITICO: gli alberi sono stati calcolati "
                  f"ma il salvataggio nello storage condiviso è fallito. Il task viene "
                  f"segnalato come fallito all'Orchestratore. Dettaglio: {e}")
            raise
        print(f"[+] [{self.worker_name}] Task salvato nello storage condiviso. Invio completato.")
        return serialized_task
    
    def exposed_predict_subset_forest(self, serialized_trees, serialized_X_test=None):
        """
        Riceve un sottoinsieme di alberi serializzati dall'Orchestratore e calcola 
        le predizioni parziali sui dati di test sfruttando il C nativo di Scikit-Learn.
        """
        print(f"\n[WORKER RPC] Ricevuta richiesta di inferenza parziale...")
        
        # 1. Ricostruiamo gli alberi inviati dal Master
        trees = pickle.loads(serialized_trees)
        print(f"[{self.worker_name}] Decodificati {len(trees)} alberi per il calcolo.")

        # 2. Gestione asimmetrica Centralizzato vs Federato
        if serialized_X_test is not None:
            if self._cached_X_test_bytes != serialized_X_test:
                self._cached_X_test_bytes = serialized_X_test
                self._cached_X_eval = pickle.loads(serialized_X_test)
                print(f"[{self.worker_name}] Decodificato il testing set centralizzato (Shape: {self._cached_X_eval.shape}).")
            else:
                print(f"[{self.worker_name}] Utilizzo del testing set centralizzato già in cache (Shape: {self._cached_X_eval.shape}).")
            X_eval = self._cached_X_eval
        else:
            if getattr(self, 'X_test', None) is None:
                raise ValueError(
                    f"[{self.worker_name}] Errore: Nessun dataset di test locale trovato in memoria. "
                    f"Esegui prima il round di addestramento federato."
                )
            X_eval = self.X_test
            print(f"[{self.worker_name}] Utilizzo del testing set federato locale (Shape: {X_eval.shape}).")

        print(f"[{self.worker_name}] Avvio inferenza nativa lineare su {len(trees)} alberi...")
        sub_predictions = [tree.predict(X_eval) for tree in trees]
            
        print(f"[+] [{self.worker_name}] Calcolo predizioni completato per {len(trees)} alberi.")
        return pickle.dumps(sub_predictions)
    

    def _get_task_storage_paths(self, source_info: str, base_seed: int, num_trees: int):
        """
        Genera i percorsi per lo storage condiviso basandosi sul TASK.
        Estrae il job_id dal source_info per evitare collisioni tra job diversi.
        """
        # Estrazione sicura del job_id dal path del file (funziona sia per S3 che locale)
        filename = os.path.basename(source_info) # es: shared_train_12345.csv
        job_id = filename.replace("shared_train_", "").replace(".csv", "")
        
        local_dir = os.path.join("./.local_storage", "trained_tasks")
        base_name = f"task_{job_id}_seed_{base_seed}_trees_{num_trees}"
        local_meta_path = os.path.join(local_dir, base_name + ".meta.json")
        local_bin_path = os.path.join(local_dir, base_name + ".bin")
        
        s3_bucket = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")
        s3_key = f"tasks/{job_id}/task_seed_{base_seed}_trees_{num_trees}.pkl"
        
        return local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key
    
    def _load_task_from_shared_storage(self, source_info: str, base_seed: int, num_trees: int) -> bytes:
        """Tenta di recuperare i byte serializzati dell'INTERO TASK dallo storage condiviso."""
        local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key = self._get_task_storage_paths(source_info, base_seed, num_trees)
        
        if self.environment == "local":
            if os.path.exists(local_bin_path):
                try:
                    
                    with open(local_bin_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    print(f"[{self.worker_name}] Errore durante la lettura del task binario locale: {e}")
        else:
                       
            try:
                
                s3_client = boto3.client("s3")
                response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
                print(f"[{self.worker_name}] [TASK HIT] Trovato task su S3: s3://{s3_bucket}/{s3_key}")
                return response['Body'].read()
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    print(f"[{self.worker_name}] Errore S3 per il task seed {base_seed}: {e}")
            except Exception as e:
                print(f"[{self.worker_name}] Errore imprevisto nel recupero del task da S3: {e}")
                
        return None
    
    def _save_task_to_shared_storage(self, source_info: str, base_seed: int, num_trees: int, serialized_trees_bytes: bytes):
        """Persiste in modo atomico i byte dell'intero TASK nello storage condiviso."""
        local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key = self._get_task_storage_paths(source_info, base_seed, num_trees)
        
        if self.environment == "local":
            try:
                
                os.makedirs(local_dir, exist_ok=True)
                tmp_bin_path = local_bin_path + ".tmp"
                with open(tmp_bin_path, "wb") as f:
                    f.write(serialized_trees_bytes)
                os.replace(tmp_bin_path, local_bin_path)
                
                tmp_meta = local_meta_path + ".tmp"
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump({
                        "base_seed": base_seed,
                        "num_trees": num_trees,
                        "size_bytes": len(serialized_trees_bytes),
                        "timestamp": time.time()
                    }, f, indent=2)
                # Sostituzione atomica per prevenire corruzioni di file
                os.replace(tmp_meta, local_meta_path)
                print(f"[{self.worker_name}] [TASK STORAGE] Task {base_seed} salvato nello storage locale condiviso.")
            except Exception as e:
                print(f"[{self.worker_name}] Errore nel salvataggio del task JSON locale: {e}")
                raise
        else:
            # Ambiente AWS: Scrittura diretta del payload binario su S3
            size_mb = len(serialized_trees_bytes) / (1024 ** 2)
            print(f"[{self.worker_name}] [TASK STORAGE] Avvio upload task su S3 "
                  f"({size_mb:.1f} MB, bucket: {s3_bucket}, key: {s3_key})...")
            start_ts = time.time()
            try:
                s3_client = boto3.client("s3")
                from boto3.s3.transfer import TransferConfig
                import io
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=s3_key,
                    Body=serialized_trees_bytes  # Passi direttamente i byte, senza io.BytesIO
                )
                elapsed = time.time() - start_ts
                print(f"[{self.worker_name}] [TASK STORAGE] Task {base_seed} salvato su S3 "
                      f"in {elapsed:.1f}s ({size_mb:.1f} MB).")
            except Exception as e:
                elapsed = time.time() - start_ts
                print(f"[{self.worker_name}] Errore nel caricamento del task su S3 dopo {elapsed:.1f}s: {e}")
                raise