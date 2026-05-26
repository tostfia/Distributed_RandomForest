import os
import json
import time

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        super().__init__(
            orchestrator_name="Orchestrator-Centralizzato-A",
            queue_name="centralized_queue",
            environment=environment
        )

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        print(f"   [CENTRALIZED-CORE] Calcolo alberi Random Forest da {start_alberi} a {target_alberi} (Seed: {seed})...")
        time.sleep(5)


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