from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import numpy as np
from rpyc import Service
import rpyc
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.factory import get_aws_services
from src.dataset.dataset_dao import *
from src.dataset.dataset_dao_factory import *


#addestramento di un singolo albero
def _train_single_tree_processor(args):
    X,y, tree_seed, max_depth,max_samples, bootstrap, tree_class = args
    np.random.seed(tree_seed)
    n_samples = X.shape[0]

    if bootstrap:
        size = max_samples if max_samples is not None else n_samples
        indices = np.random.choice(n_samples, size=size, replace=True)
        X_train, y_train = X[indices], y[indices]
    else: 
        X_train, y_train = X, y
   
    tree = tree_class(splitter="best", max_depth=max_depth) 
    tree.fit(X_train, y_train)
    return tree

class  BaseWorker(Service,ABC): 
    def __init__(self, worker_name: str, queue_name: str, environment: str, url_dataset: str, tree_class_reference, max_samples= None, bootstrap: bool = True):
        self.worker_name  = worker_name
        self.environment = environment
        self.queue_name = queue_name
        self.url_dataset = url_dataset
        self.tree_class_reference = tree_class_reference
        self.max_samples = max_samples
        self.bootstrap = bootstrap

    def _get_my_private_ip(self) -> str:
        "Rilevamento automaticamente l'IP privato del nodo corrente"
        if self.environment == "AWS":
            pass
            return "0.0.0.0"
        return "127.0.0.1"

    def start_server(self, port: int, explicit_host: str = None):

        "Avvio il server RPyC"
        if self.environment == "aws":
            pass #Agginta del codice da implementare

        host_to_register = explicit_host if explicit_host else "127.0.0.1"
        host_to_bind = host_to_register

        ServiceRegistry.register_worker(worker_name = self.worker_name, host = host_to_register, port = port)

        ServiceRegistry.start_heartbeat(node_name = self.worker_name, node_type="worker", host=host_to_register, port=port)



    def on_connect(self, conn):
        print(f"[+] Orchestrator connesso: {conn._config['peer']}")

    def on_disconnect(self, conn):
        print(f"[-] Orchestrator disconesso")

    @abstractmethod
    def _load_data(self, source_info):

        pass

    @abstractmethod
    def _get_tree_class(self):

        pass

    @rpyc.exposed
    def train_subset_forest(self, source_info, num_trees,base_seed, max_depth=None):

        print(f"[*] Richiesta ricevuta su BaseWorker: generazione di {num_trees} alberi.")

        #1. Recupero dati e classe dell'albero dalle classi figlie
        X,y = self._load_data(source_info)
        tree_class = self._get_tree_class()

        #2. Generazione dei parametri per i singoli core
        worker_tasks=[]
        for i in range(num_trees):
            tree_seed = base_seed + i
            worker_tasks.append((X,y, tree_seed, max_depth, self.max_samples, self.bootstrap, tree_class))

        #3. parallelizzazione sui core della CPU locale
        with Pool() as pool: 
            local_trees= pool.map(_train_single_tree_processor, worker_tasks)

        print(f"[+] Addestramento di {num_trees} alberi completato con successo")
        return local_trees
    
    

