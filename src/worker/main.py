import os
import sys
import multiprocessing

from src.shared.config import SystemConfig
from src.worker.centralizedWorker import CentralizedWorker
from src.worker.federatedWorker import FederatedWorker
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
try:
    multiprocessing.set_start_method('spawn', force=True)
    print("[MULTIPROCESSING] Metodo 'spawn' attivato per evitare crash RPyC.")
except RuntimeError:
    pass


def main():
    # 1. Carichiamo la configurazione centrale dal file .env
    cfg = SystemConfig()
    mode = cfg.mode
    environment = cfg.env

    # 2. Gestiamo i parametri da terminale minimi (solo Nome e Porta per evitare conflitti di rete)
    if len(sys.argv) < 3:
        print("\n[ERRORE] Parametri Insufficienti per l'istanza di rete.")
        print("Uso corretto: python -m src.worker.run_worker <NOME_WORKER> <PORTA> [CLASSIFIER/REGRESSOR]")
        sys.exit(1)
    
    worker_name = sys.argv[1]
    
    try:
        port = int(sys.argv[2])
    except ValueError:
        print("[ERRORE] La porta deve essere un numero intero valido.")
        sys.exit(1)
        
    # Tipo di albero opzionale (default: classifier)
    tree_type = sys.argv[3].lower() if len(sys.argv) > 3 else "classifier"

    # Recuperiamo l'IP di rete (utile se impostato da Docker o variabili di sistema)
    host = os.environ.get("RPC_HOST", "127.0.0.1")

    print("=====================================================")
    print(f"        INIZIALIZZAZIONE NODO WORKER CLUSTER         ")
    print(f"  • Nome Nodo:           {worker_name}")
    print(f"  • Porta RPC:           {port}")
    print(f"  • Modalità (.env):     {mode.upper()}")
    print(f"  • Ambiente (.env):     {environment.upper()}")
    print(f"  • Tipo Algoritmo:      {tree_type.upper()}")
    print("=====================================================\n")

    # 3. Preparazione dei parametri puliti (senza environment e url_dataset)
    common_params = {
        "worker_name": worker_name,
        "queue_name": "centralized_queue" if mode == "centralized" else "federated_queue",
        "tree_class_reference": DecisionTreeClassifier if tree_type == "classifier" else DecisionTreeRegressor,
        "max_samples": None,  
        "bootstrap": True ,  
    }

    # 4. Istanziamo il Worker corretto in base a TRAINING_MODE del file .env
    if mode == "centralized":
        print(f"[*] Istanziazione in corso: comportamento CENTRALIZZATO per {worker_name}")
        worker = CentralizedWorker(**common_params, target_column="Label")
    elif mode == "federated":
        print(f"[*] Istanziazione in corso: comportamento FEDERATO per {worker_name}")
        worker = FederatedWorker(**common_params, target_column="Label", tree_type=tree_type)
    else:
        print(f"[ERRORE] TRAINING_MODE '{mode}' non valida nel file .env. Scegliere 'centralized' o 'federated'.")
        sys.exit(1)
    
    print(f"[+] Avvio Server RPyC per il worker {worker_name} sulla porta {port} (Host: {host})...")
    
    try:
        # Avvia il server RPC (gestisce internamente heartbeat loop e ServiceRegistry)
        worker.start_server(port=port, explicit_host=host) 
    except Exception as e:
        print(f"[ERRORE CRITICO ENCOUNTERED]: {e}")


if __name__ == "__main__":
    main()