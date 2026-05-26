import os
import json
import time

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator

class FederatedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        # Recuperiamo il Process ID per generare un nome univoco per ogni replica federata
        pid = os.getpid()
        super().__init__(
            orchestrator_name=f"Orchestrator-Federato-{pid}",
            queue_name="federated_queue",
            environment=environment
        )

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        """Implementazione del coordinamento e dell'aggregazione (Federated Averaging)."""
        
        step_dim = target_alberi - start_alberi
        round_num = (start_alberi // step_dim) + 1
        
        print(f"   [{self.orchestrator_name}] === AVVIO ROUND {round_num} ===")
        print(f"   [{self.orchestrator_name}] -> Distribuzione calcolo alberi ({start_alberi} a {target_alberi}) ai nodi remoti... (Seed: {seed})")
        
        # 1. AUMENTATO A 45 SECONDI: Simulazione del calcolo sui nodi per darti il tempo di killare il processo
        time.sleep(45) 
        
        # 2. Simulazione della fase di aggregazione dei modelli locali (Federated Averaging)
        print(f"   [{self.orchestrator_name}] -> Ricezione dei pesi locali dai nodi completata.")
        print(f"   [{self.orchestrator_name}] -> Aggregazione e generazione del Modello Globale per il Round {round_num} eseguita.")
        time.sleep(1)


if __name__ == "__main__":
    # Lettura dinamica dell'ambiente configurato nel config.json locale
    env = "local"
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                env = config.get("environment", "local")
        except Exception:
            pass  # Fallback su local se il file è corrotto o mancante
            
    # Istanziamo e avviamo l'orchestratore federato
    orchestrator = FederatedOrchestrator(environment=env)
    orchestrator.start()