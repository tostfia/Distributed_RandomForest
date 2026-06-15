from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import numpy as np
import rpyc
from rpyc import Service, ThreadedServer
import threading
import time
import pickle 

from src.shared.binding.serviceregistry import ServiceRegistry

# Aggiungi questo metodo astratto in BaseWorker
@abstractmethod
def is_regression(self):
    pass

# addestramento di un singolo albero
def _train_single_tree_processor(args):
    X, y, tree_seed, max_depth, max_samples, bootstrap, tree_class = args
    np.random.seed(tree_seed)
    n_samples = X.shape[0]

    if bootstrap:
        size = int(max_samples * n_samples) if max_samples else n_samples
        indices = np.random.choice(n_samples, size=size, replace=True)
        X_train, y_train = X[indices], y[indices]
    else: 
        X_train, y_train = X, y
   
    tree = tree_class(splitter="best", max_depth=max_depth) 
    tree.fit(X_train, y_train)
    return tree


class BaseWorker(Service, ABC): 
    def __init__(self, worker_name: str, queue_name: str, environment: str, url_dataset: str, tree_class_reference, max_samples=None, bootstrap: bool = True):
        super().__init__() 
        self.worker_name = worker_name
        self.environment = environment
        self.queue_name = queue_name
        self.url_dataset = url_dataset
        self.tree_class_reference = tree_class_reference
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self._stop_heartbeat = None

    def _get_my_private_ip(self) -> str:
        if self.environment == "AWS":
            return "0.0.0.0"
        return "127.0.0.1"

    def start_server(self, port: int, explicit_host: str = None):
        if self.environment == "aws":
            pass 

        host_to_register = explicit_host if explicit_host else "127.0.0.1"
        host_to_bind = "0.0.0.0" if self.environment.lower() == "local" else host_to_register

        ServiceRegistry.register_worker(worker_name=self.worker_name, host=host_to_register, port=port)

        self._stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, args=(self._stop_heartbeat, 10), daemon=True)
        heartbeat_thread.start()
        print(f"[+] [{self.worker_name}] Thread avviato sul nodo")

        protocol_config = {
            'allow_public_attr': True,
            'allow_pickle': True,
            'sync_request_timeout': 600, 
            'keepalive': True
        }

        server = ThreadedServer(self, hostname=host_to_bind, port=port, protocol_config=protocol_config)

        print("\n ==============================================")
        print(f"\n [Modalità Locale] SERVER IN ASCOLTO: {self.worker_name.upper()}")
        print(f"\n [Modalità Locale] Indirizzo di ascolto: {host_to_bind}:{port}")
        print("\n ==============================================")

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
            try:
                ServiceRegistry.deregister_worker(self.worker_name)
                print(f"\n[+] [{self.worker_name}] Server arrestato e worker deregistrato.") 
            except Exception as e:
                print(f"\n[!] [{self.worker_name}] Errore durante la deregistrazione: {str(e)}")

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
        print(f"[+] Orchestratore connesso: {peer_info}")

    def on_disconnect(self, conn):
        print(f"[-] Orchestratore disconnesso")

    @abstractmethod
    def _load_data(self, source_info):
        pass

    @abstractmethod
    def _get_tree_class(self):
        pass

    def exposed_train_subset_forest(self, source_info, num_trees, base_seed, max_depth=None):
        # BANNER VISIVO DI VERIFICA VERSIONE
        print("\n=============================================================")
        print(f" [WORKER BANNER] OK! STO ESEGUENDO LA NUOVA VERSIONE CON PICKLE! Alberi: {num_trees}")
        print("=============================================================\n")

        # 1. Recupero dati e classe dell'albero dalle classi figlie
        X, y = self._load_data(source_info)
        tree_class = self._get_tree_class()

        # 2. Generazione dei parametri per i singoli core
        worker_tasks = []
        for i in range(num_trees):
            tree_seed = base_seed + i
            worker_tasks.append((X, y, tree_seed, max_depth, self.max_samples, self.bootstrap, tree_class))

        # 3. Ottimizzazione anti-crash per il multiprocessing in Docker
        if num_trees == 1:
            print("[WORKER] Ottimizzazione: 1 solo albero richiesto. Esecuzione diretta senza Pool per stabilità.")
            local_trees = [_train_single_tree_processor(worker_tasks[0])]
        else:
            print(f"[WORKER] Avvio parallelizzazione controllata con Pool (max 2 processi) per {num_trees} alberi.")
            with Pool(processes=2) as pool: 
                local_trees = pool.map(_train_single_tree_processor, worker_tasks)

        print(f"[+] Addestramento di {num_trees} alberi completato con successo. Serializzazione in byte...")
        
        # Ritorniamo i dati convertiti in byte nativi (nessun oggetto proxy RPyC)
        return pickle.dumps(local_trees)