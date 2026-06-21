from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import os
import numpy as np
import rpyc
from rpyc import Service, ThreadedServer
import threading
import time
import pickle 
from concurrent.futures import ThreadPoolExecutor

from src.shared.config import SystemConfig  # <-- INCLUSO CONFIG CENTRALE
from src.shared.binding.serviceregistry import ServiceRegistry

_child_X  = None
_child_y  = None
def _init_child_process(X, y):
    """Inizializza il processo figlio salvando i dati in memoria una sola volta."""
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
        """Determina l'IP corretto per il binding di rete in base all'ambiente."""
        if self.environment == "aws":
            return "0.0.0.0"  # Su AWS ascolta su tutte le interfacce del container/istanza
        return "127.0.0.1"

    def start_server(self, port: int, explicit_host: str = None):
        print(f"\n[{self.worker_name}] Inizializzazione Server RPC in ambiente {self.environment.upper()}...")

        advertise_host = os.environ.get("RPC_ADVERTISE_HOST", None)
        # Gestione degli host coerente con l'ambiente del file .env
        if self.environment == "aws":
            host_to_bind = "0.0.0.0"
            host_to_register = advertise_host if advertise_host else (explicit_host if explicit_host else "0.0.0.0")
        else:
            if advertise_host:
                host_to_bind = "0.0.0.0"
                host_to_register = advertise_host
            else:
                host_to_bind = explicit_host if explicit_host else "127.0.0.1"
                host_to_register = explicit_host if explicit_host else "127.0.0.1"

        # Registrazione del Worker sul Service Registry (Mock o DynamoDB gestito in automatico)
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
                print(f"[WORKER] Creazione nuovo Pool di processi per {num_trees} alberi...")
                self._cached_pool = Pool(processes=min(num_trees, os.cpu_count()), initializer=_init_child_process, initargs=(X, y))
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
        le predizioni parziali sui dati di test.
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
            # Caso Federato: i dati di test risiedono localmente sul Worker
            if getattr(self, 'X_test', None) is None:
                raise ValueError(
                    f"[{self.worker_name}] Errore: Nessun dataset di test locale trovato in memoria. "
                    f"Esegui prima il round di addestramento federato."
                )
            X_eval = self.X_test
            print(f"[{self.worker_name}] Utilizzo del testing set federato locale (Shape: {X_eval.shape}).")

        print(f"[{self.worker_name}] Avvio inferenza parallela su {len(trees)} alberi...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            sub_predictions = list(executor.map(lambda tree: tree.predict(X_eval), trees))
            
        print(f"[+] [{self.worker_name}] Calcolo predizioni completato per {len(trees)} alberi.")
        
        # Restituiamo la matrice parziale all'Orchestratore
        return pickle.dumps(sub_predictions)