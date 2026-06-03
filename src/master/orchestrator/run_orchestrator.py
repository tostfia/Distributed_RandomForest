import sys
import os

from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator

def main(): 
    if len(sys.argv) <3:
        print("\n[ERRORE] Parametri Insufficienti")
        print("Uso corretto: python -m src.master.orchestrator.run_orchestrator <MODO (centralized/federated)> <ENVIRONMENT (local/aws)>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    environment = sys.argv[2].lower()

    if environment not in ["local", "aws"]:
        print("\n[ERRORE] Environment non valido. Scegliere 'local' o 'aws'.")
        sys.exit(1)

    if mode == "centralized":
        print(f"\n[INFO] Avvio Orchestratore Centralizzato in ambiente '{environment}'...")
        orchestrator = CentralizedOrchestrator(environment=environment)
    elif mode == "federated":
        print(f"\n[INFO] Avvio Orchestratore Federato in ambiente '{environment}'...")
        orchestrator = FederatedOrchestrator(environment=environment)
    else:
        print("\n[ERRORE] Modalità non valida. Scegliere 'centralized' o 'federated'.")
        sys.exit(1)

    try: 
        orchestrator.start()
    except KeyboardInterrupt:
        print("\n[INFO] Orchestratore interrotto manualmente. Uscita in corso...")
        sys.exit(0)


if __name__ == "__main__":
    main()