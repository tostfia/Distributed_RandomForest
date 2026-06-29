import time
import threading
from testing.scenarios import BaseTestScenario


class FaultToleranceScenario(BaseTestScenario):
    """Copre lo Scenario 5: Sperimentazione della Tolleranza ai Guasti (Failover)."""
    def run(self) -> dict:
        ft_cfg = self.config["fault_tolerance"]
        print("\n--- [SCENARIO 5] Sperimentazione della Tolleranza ai Guasti (Kill Worker) ---")
        
        def kill_worker_target():
            time.sleep(ft_cfg["kill_worker_after_seconds"])
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker...")
            # Logica per simulare il crash di un worker. 
            # Esempio: chiudere forzatamente una delle connessioni RPyC nell'orchestratore
            if hasattr(self.orchestrator, "worker_channels") and self.orchestrator.worker_channels:
                try:
                    target_worker = list(self.orchestrator.worker_channels.keys())[0]
                    self.orchestrator.worker_channels[target_worker].close()
                    print(f"[TEST TRIGGER] Connessione con il Worker {target_worker} interrotta.")
                except Exception as e:
                    print(f"[TEST TRIGGER ERRORE] Impossibile chiudere il worker: {e}")

        # Avvia il thread killer in background che agirà durante l'addestramento
        killer_thread = threading.Thread(target=kill_worker_target)
        killer_thread.daemon = True
        
        payload = {
            "job_id": f"test_fault_{int(time.time())}",
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {"n_estimators": 60, "max_depth": 5, "tree_type": self.config["selected_task"]}
        }
        
        killer_thread.start()
        
        start_time = time.perf_counter()
        # _execute_training_step intercetterà l'eccezione RPC del worker chiuso, 
        # reinserirà il chunk nella coda e terminerà l'addestramento grazie agli altri worker.
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=60, seed=42)
        duration = time.perf_counter() - start_time
        
        status = "SUCCESS" if num_trees >= ft_cfg["expected_min_trees"] else "FAILED"
        return {
            "status": status,
            "duration_seconds": duration,
            "trees_built": num_trees
        }