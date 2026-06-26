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
        print(f"[INFO] Istanzio l'Orchestratore Centralizzato...")
        orchestrator = CentralizedOrchestrator(orchestrator_name=orchestrator_name)
        
    elif mode == "federated":
        print(f"[INFO] Modalità Federata rilevata.")
        
        # Eseguiamo il bootstrap dei file CSV LOCALI SOLO se siamo in ambiente di sviluppo "local"
        if environment == "local":
            try:
                workers_attivi = ServiceRegistry.get_available_workers("local")
                num_workers = len(workers_attivi)
                if num_workers > 0:
                    print(f"[BOOTSTRAP] Rilevati dinamicamente {num_workers} worker attivi nel ServiceRegistry: {workers_attivi}")
                else:
                    # Fallback: se l'orchestratore parte un secondo prima dei worker, il registry potrebbe essere vuoto
                    num_workers = int(getattr(cfg, "num_workers", 3))
                    print(f"[BOOTSTRAP INFO] Nessun worker ancora registrato. Uso il fallback da configurazione: {num_workers}")
            except Exception as e:
                num_workers = int(getattr(cfg, "num_workers", 3))
                print(f"[BOOTSTRAP WARN] Impossibile leggere il ServiceRegistry ({e}). Fallback su configurazione: {num_workers}")
            
            # --- BOOTSTRAP DATASET REALE (Solo Local) ---
            try:
                
                print("[BOOTSTRAP] Generazione Shard per DATASET REALE...")
                
                # ALLINEAMENTO BASELINE: Identifichiamo la cartella cache dei dati
                data_folder = getattr(cfg, "dataset_path", None)
                if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
                    data_folder = "./dataset_cache" if os.path.exists("./dataset_cache") else "./data"

                #print(f" • Cartella sorgente identificata per bootstrap federato: '{data_folder}'")
                
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