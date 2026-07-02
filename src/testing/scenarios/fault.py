import time
import threading
import os
from src.testing.scenarios.base import BaseTestScenario


class FaultToleranceScenario(BaseTestScenario):
    """Copre lo Scenario 5: Sperimentazione della Tolleranza ai Guasti (Kill Worker)."""
    

    def run(self) -> dict:
        ft_cfg = self.config.get("fault_tolerance", {})
        kill_delay = ft_cfg.get("kill_worker_after_seconds", 10)
        
        def kill_worker_local():
            time.sleep(kill_delay)
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker (Locale)...")
            try:
                if hasattr(self.orchestrator, "connessioni_attive") and self.orchestrator.connessioni_attive:
                    target_conn = self.orchestrator.connessioni_attive[0]
                    target_conn.close()
                    print("[TEST TRIGGER] Connessione RPyC interrotta con successo!")
            except Exception as e:
                print(f"[TEST ERRORE] {e}")

        threading.Thread(target=kill_worker_local, daemon=True).start()
        
        start_time = time.perf_counter()
        payload = self._build_payload()
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=60, seed=42)
        duration = time.perf_counter() - start_time
        
        return {
            "scenario_description": "Crash improvviso Worker su thread/processi Python locali.",
            "execution_mode": "local",
            "status": "SUCCESS" if num_trees >= ft_cfg.get("expected_min_trees", 50) else "FAILED",
            "trees_built": num_trees, 
            "duration_seconds": round(duration, 2)
        }

    def _build_payload(self):
        return {
            "job_id": f"test_fault_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": {"n_estimators": 60, "max_depth": 5, "tree_type": self.config.get("selected_task", "classifier")}
        }