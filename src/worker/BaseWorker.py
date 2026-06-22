from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import os
import socket
import numpy as np
from rpyc import Service, ThreadedServer
import threading
import time
import pickle 
from concurrent.futures import ThreadPoolExecutor

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

# Addestramento di un singolo albero (resta fuori dalla classe per il multiprocessing)
def _train_single_tree_processor(args):
    """Esegue l'addestramento prelevando X e y dalla memoria globale del processo."""
    global _child_X, _child_y
    tree_seed, max_depth, max_samples, bootstrap, tree_class = args
    np.random.seed(tree_seed)
    
    # Preleviamo la shape direttamente dalla memoria condivisa del processo figlio
    n_samples = _child_X.shape[0]

    if bootstrap:
        size = int(max_samples * n_samples) if max_samples else n_samples
        indices = np.random.choice(n_samples, size=size, replace=True)
        X_train, y_train = _child_X[indices], _child_y[indices]
    else: 
        X_train, y_train = _child_X, _child_y
   
    tree = tree_class(splitter="best", max_depth=max_depth) 
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

    def exposed_train_subset_forest(self, source_info, num_trees, base_seed, max_depth=None):
        print("\n=============================================================")
        print(f" [WORKER RPC] Richiesta elaborazione foresta parziale | Alberi: {num_trees}")
        print("=============================================================\n")

        # 1. Recupero dati e classe dell'albero dalle classi figlie
        X, y = self._load_data(source_info)
        tree_class = self._get_tree_class()

        # 2. CALCOLO DINAMICO DEI CORE (Locale vs AWS)
        totale_core_macchina = os.cpu_count() or 1
        
        if self.environment != "aws":
            try:
                # Interroghiamo il ServiceRegistry usando l'ambiente del worker
                workers_attivi = ServiceRegistry.get_available_workers(self.environment)
                num_workers = max(1, len(workers_attivi))
                
                # Formula di bilanciamento per i tuoi 8 core in locale
                core_disponibili_rete = max(1, totale_core_macchina - 1)
                allocated_cores = max(1, int(core_disponibili_rete / num_workers))
                print(f"[{self.worker_name}] [LOG LOCALE] Rilevati {num_workers} worker attivi sulla macchina.")
                print(f"[{self.worker_name}] [LOG LOCALE] Allocazione dinamica: {allocated_cores} processi per questo pool.")
            except Exception as e:
                print(f"[!] Errore lettura ServiceRegistry, fallback su N-1: {e}")
                allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina
        else:
            # Su AWS ogni worker ha la sua macchina isolata, usa la regola standard N-1
            allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina

        # 3. Ottimizzazione anti-crash per il multiprocessing in Docker
        if num_trees == 1:
            print("[WORKER] Ottimizzazione: 1 solo albero richiesto. Esecuzione diretta senza Pool.")
            direct_task  = (base_seed, max_depth, self.max_samples, self.bootstrap, tree_class)

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
                worker_tasks.append((seed, max_depth, self.max_samples, self.bootstrap, tree_class))
            local_trees = self._cached_pool.map(_train_single_tree_processor, worker_tasks)

        print(f"[+] Calcolo di {num_trees} alberi completato. Invio in corso via pickle...")
        return pickle.dumps(local_trees)
    
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