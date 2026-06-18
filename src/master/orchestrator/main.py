import sys

from src.shared.config import SystemConfig
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator

def main(): 
    # 1. Carichiamo l'intera configurazione centralizzata dal file .env
    cfg = SystemConfig()
    
    mode = getattr(cfg, "mode", "centralized").strip().lower()
    environment = cfg.env

    print("=====================================================")
    print(f"       INIZIALIZZAZIONE NODO MASTER CLUSTER          ")
    print(f"  • Modalità operativa (.env):  {mode.upper()}")
    print(f"  • Ambiente cloud/local (.env): {environment.upper()}")
    print("=====================================================\n")

    # 2. Istanziamo l'orchestratore corretto basandoci solo sul file .env
    if mode == "centralized":
        print(f"[INFO] Istanzio l'Orchestratore Centralizzato...")
        orchestrator = CentralizedOrchestrator()
    elif mode == "federated":
        print(f"[INFO] Istanzio l'Orchestratore Federato...")
        orchestrator = FederatedOrchestrator()
    else:
        print(f"\n[ERRORE] SYS_MODE '{mode}' non valida nel file .env. Scegliere 'centralized' o 'federated'.")
        sys.exit(1)

    try: 
        # Avvia il ciclo di vita (polling SQS, heartbeat, failover)
        orchestrator.start()
    except KeyboardInterrupt:
        print("\n[INFO] Ricevuto segnale di terminazione. Uscita dall'orchestratore in corso...")
        sys.exit(0)


if __name__ == "__main__":
    main()