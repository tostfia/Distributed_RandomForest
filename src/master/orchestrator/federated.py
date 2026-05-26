import os
import json
import time

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator

class FederatedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        # Passiamo alla classe madre i parametri specifici per la modalità Federata
        super().__init__(
            orchestrator_name="Orchestrator-Federato-Master",
            queue_name="federated_queue",
            environment=environment
        )

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        """Implementazione del coordinamento e dell'aggregazione (Federated Averaging)."""
        
        # Calcoliamo dinamicamente l'indice del round corrente in base agli alberi fatti
        step_dim = target_alberi - start_alberi
        round_num = (start_alberi // step_dim) + 1
        
        print(f"   [FEDERATED-MASTER] === AVVIO ROUND {round_num} ===")
        print(f"   [FEDERATED-MASTER] -> Distribuzione calcolo alberi ({start_alberi} a {target_alberi}) ai nodi remoti... (Seed: {seed})")
        
        # 1. Simulazione del tempo di calcolo in parallelo sui nodi
        time.sleep(3) 
        
        # 2. Simulazione della fase di aggregazione dei modelli locali (Federated Averaging)
        print(f"   [FEDERATED-MASTER] -> Ricezione dei pesi locali dai nodi completata.")
        print(f"   [FEDERATED-MASTER] -> Aggregazione e generazione del Modello Globale per il Round {round_num} eseguita.")
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