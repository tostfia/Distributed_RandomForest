import time
import threading
import os
from src.testing.scenarios.base import BaseTestScenario


class FaultToleranceScenario(BaseTestScenario):
    """Copre lo Scenario 5: Sperimentazione della Tolleranza ai Guasti (Kill Worker)."""
    

    def run(self) -> dict:
        ft_cfg = self.config.get("fault_tolerance", {})
        kill_delay = ft_cfg.get("kill_worker_after_seconds", 10)
        task_type = self.config.get("selected_task", "classifier")
        if task_type == "classifier":
            target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
        else:
            target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)
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
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
        duration = time.perf_counter() - start_time
        
        return {
            "scenario_description": "Crash improvviso Worker su thread/processi Python locali.",
            "execution_mode": "local",
            "status": "SUCCESS" if num_trees == target_trees else "FAILED",
            "trees_built": num_trees, 
            "duration_seconds": round(duration, 2)
        }

    def _build_payload(self):
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        return {
            "job_id": f"test_fault_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": hp,
        }