import sys
import os
import socket
import signal as os_signal

from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator

def main(): 
    cfg = SystemConfig()
    mode = getattr(cfg, "mode", "centralized").strip().lower()
    environment = cfg.env

    ec2_id = os.environ.get("EC2_ID", "Locale")
    hostname = socket.gethostname()

    orchestrator_index = os.environ.get("ORCHESTRATOR_INDEX", "1")
    orchestrator_name = f"Orchestrator-{ec2_id}-{mode}-{hostname}-{orchestrator_index}"

    def _graceful_shutdown(signum, frame):
        print(f"[SHUTDOWN] Ricevuto segnale {signum}, deregistro {orchestrator_name}...")
        ServiceRegistry.deregister_orchestrator(orchestrator_name)
        sys.exit(0)

    os_signal.signal(os_signal.SIGTERM, _graceful_shutdown)

    print("=====================================================")
    print(f"       INIZIALIZZAZIONE NODO MASTER CLUSTER          ")
    print(f"  • Modalità operativa (.env):  {mode.upper()}")
    print(f"  • Ambiente cloud/local (.env): {environment.upper()}")
    print("=====================================================\n")

    if mode == "centralized":
        num_workers = int(os.environ.get("NUM_WORKERS", 3))
        print(f" Numero di worker configurato da .env: {num_workers}")
        print(f"[INFO] Istanzio l'Orchestratore Centralizzato...")
        orchestrator = CentralizedOrchestrator(orchestrator_name=orchestrator_name)
        orchestrator.queue_name = cfg.sqs_centralized_queue
        print(f"[CONFIG] Coda associata: '{orchestrator.queue_name}'")
        
    elif mode == "federated":
        print(f"[INFO] Modalità Federata rilevata.")
        num_workers = int(os.environ.get("NUM_WORKERS", 3))
        print(f" Numero di worker configurato da .env: {num_workers}")
        print(f"[INFO] Istanzio l'Orchestratore Federato...")
        orchestrator = FederatedOrchestrator(orchestrator_name=orchestrator_name, num_workers=num_workers)
        orchestrator.queue_name = cfg.sqs_federated_queue
        print(f"[CONFIG] Coda associata: '{orchestrator.queue_name}'")
        
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