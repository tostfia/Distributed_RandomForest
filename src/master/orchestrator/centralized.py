import os
import json
import time

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        # Recuperiamo il Process ID per distinguere le repliche nei log
        pid = os.getpid()
        super().__init__(
            orchestrator_name=f"Orchestrator-Centralizzato-{pid}",
            queue_name="centralized_queue",
            environment=environment
        )

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        print(f"   [{self.orchestrator_name}] Calcolo alberi Random Forest da {start_alberi} a {target_alberi} (Seed: {seed})...")
        # AUMENTIAMO IL TEMPO A 45 SECONDI PER DARCI IL TEMPO DI REAGIRE
        time.sleep(45)


if __name__ == "__main__":
    env = "local"
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                env = json.load(f).get("environment", "local")
        except Exception:
            pass
            
    orchestrator = CentralizedOrchestrator(environment=env)
    orchestrator.start()