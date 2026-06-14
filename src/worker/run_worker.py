import os
import sys
from sklearn.tree import DecisionTreeClassifier

from src.worker.centralizedWorker import CentralizedWorker
from src.worker.federatedWorker import FederatedWorker
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
    print("[MULTIPROCESSING] Metodo 'spawn' attivato per evitare crash RPyC.")
except RuntimeError:
    pass

def main():
    if len(sys.argv) < 5:
        print("\n[ERRORE] Parametri Insufficienti")
        print("Uso corretto: python -m src.worker.run_worker <NOME_WORKER> <PORTA> <MODO (centralized/federated)> <ENVIRONMENT (local/aws)>")
        sys.exit(1)
    
    worker_name = sys.argv[1]
    mode = sys.argv[3].lower()
    environment = sys.argv[4].lower()

    try:
        host = os.environ.get("RPC_HOST", "127.0.0.1")
        port = int(sys.argv[2])
    except ValueError:
        print("[ERRORE] La porta deve essere un numero intero valido.")
        sys.exit(1)

    common_params = {
        "worker_name": worker_name,
        "queue_name": "centralized_queue" if mode == "centralized" else "federated_queue",
        "environment": "local" if environment == "local" else "aws",
        "url_dataset": "local_source_info",  
        "tree_class_reference": DecisionTreeClassifier,
        "max_samples": None,  
        "bootstrap": True if mode == "centralized" else False,  
    }

    if mode == "centralized":
        print(f"[*] Istanziazione in corso: comportamento CENTRALIZZATO per {worker_name}")
        worker = CentralizedWorker(**common_params, target_column="Label")
    elif mode == "federated":
        print(f"[*] Istanziazione in corso: comportamento FEDERATO per {worker_name}")
        worker = FederatedWorker(**common_params, target_column="Label")
    else:
        # Corretto il baco della stringa non chiusa
        print("[ERRORE] Modalità sconosciuta. Scegli tra 'centralized' e 'federated'")
        sys.exit(1)
    
    print(f"[+] Avvio Server RPyC per il worker {worker_name} sulla porta {port} (Host registrato: {host})...")
    print(f"[DEBUG] Registrazione worker su: {host}:{port}")
    try:
        
        worker.start_server(port=port, explicit_host=host) 
    except Exception as e:
        print(f"[ERRORE CRITICO ENCOUNTERED]: {e}")


    

# Spostato fuori dal main e corretto l'operatore '=='
if __name__ == "__main__":
    main()