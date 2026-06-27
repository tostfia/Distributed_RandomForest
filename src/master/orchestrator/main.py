import sys
import os
import socket
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.federated_data_splitter import FederatedDataSplitter
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.config import SystemConfig
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator

def main(): 
    cfg = SystemConfig()
    mode = getattr(cfg, "mode", "centralized").strip().lower()
    environment = cfg.env  # "local" oppure "aws"

    ec2_id = os.environ.get("EC2_ID", "Locale")
    hostname = socket.gethostname()
    orchestrator_name = f"Orchestrator-{ec2_id}-{mode}-{hostname}"

    print("=====================================================")
    print(f"       INIZIALIZZAZIONE NODO MASTER CLUSTER          ")
    print(f"  • Modalità operativa (.env):  {mode.upper()}")
    print(f"  • Ambiente cloud/local (.env): {environment.upper()}")
    print("=====================================================\n")

    if mode == "centralized":
        num_workers = int(getattr(cfg, "num_workers", 3))
        print(f" Numero di worker configurato da .env: {num_workers}")
        print(f"[INFO] Istanzio l'Orchestratore Centralizzato...")
        orchestrator = CentralizedOrchestrator(orchestrator_name=orchestrator_name)
        
    elif mode == "federated":
        print(f"[INFO] Modalità Federata rilevata.")
        num_workers = int(getattr(cfg, "num_workers", 3))
        print(f" Numero di worker configurato da .env: {num_workers}")
        # Eseguiamo il bootstrap dei file CSV LOCALI SOLO se siamo in ambiente di sviluppo "local"
        if environment == "local":

            # --- BOOTSTRAP DATASET REALE (Solo Local) ---
            try:
                
                print("[BOOTSTRAP] Generazione Shard per DATASET REALE...")
                
                # ALLINEAMENTO BASELINE: Identifichiamo la cartella cache dei dati
                data_folder = getattr(cfg, "dataset_path", None)
                if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
                    data_folder = "./dataset_cache" if os.path.exists("./dataset_cache") else "./data"
                shards_esistenti = True
                base_cache_dir = "./workers_cache"
                for i in range(1, num_workers + 1):
                    # Controlliamo sia il formato con zero-padding (01, 02) che quello standard (1, 2)
                    dir_padded = os.path.join(base_cache_dir, f"Worker-Locale-{i:02d}")
                    dir_unpadded = os.path.join(base_cache_dir, f"Worker-Locale-{i}")
                    
                    train_p = os.path.join(dir_padded, "train_shard.csv")
                    test_p = os.path.join(dir_padded, "test_shard.csv")
                    train_up = os.path.join(dir_unpadded, "train_shard.csv")
                    test_up = os.path.join(dir_unpadded, "test_shard.csv")
                    
                    # Se per un worker mancano i file in entrambe le possibili varianti di cartella, dobbiamo ricalcolare
                    if not ((os.path.exists(train_p) and os.path.exists(test_p)) or 
                            (os.path.exists(train_up) and os.path.exists(test_up))):
                        shards_esistenti = False
                        break
                
                
                if shards_esistenti:
                    print(f"[BOOTSTRAP] Shard già presenti su disco per tutti i {num_workers} worker. Salto il ricalcolo.")
                else:
                    print("[BOOTSTRAP] Shard incompleti o assenti. Avvio Generazione Shard per DATASET REALE...")
                    data_loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=0.05, dataset_seed=123)
                    splitter = FederatedDataSplitter(target_column="Label", test_size=0.20, random_state=123)
                    
                    # Lo splitter genererà i file 'train_shard.csv' e 'test_shard.csv' nei folder dei singoli worker
                    splitter.split_and_shard(data_loader, num_workers=num_workers, environment="local")
                    print("[BOOTSTRAP OK] Shard reali distribuiti nelle cartelle locali dei Worker.")
            except Exception as e:
                print(f"[BOOTSTRAP WARN] Salto bootstrap reale locale (es. cartella o file non trovati in {data_folder}): {e}")
        else:
            # Siamo su AWS! Non facciamo nulla al boot del Master.
            # I Worker prenderanno i rispettivi shard pre-partizionati direttamente da Amazon S3.
            print(f"[INFO] Ambiente AWS rilevato. Il bootstrap locale viene saltato. "
                  f"I nodi Worker interagiranno con gli shard persistiti sul bucket S3.")

        print(f"\n[INFO] Istanzio l'Orchestratore Federato...")
        orchestrator = FederatedOrchestrator(orchestrator_name=orchestrator_name)
        
    else:
        print(f"\n[ERRORE] SYS_MODE '{mode}' non valida.")
        sys.exit(1)

    try: 
        orchestrator.start()
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Arresto anomalo dell'Orchestratore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()