import time
import threading
from src.testing.scenarios.base import BaseTestScenario


class FaultToleranceScenario(BaseTestScenario):
    """Copre lo Scenario 5: Sperimentazione della Tolleranza ai Guasti (Failover)."""
    def run(self) -> dict:
        ft_cfg = self.config["fault_tolerance"]
        print("\n--- [SCENARIO 5] Sperimentazione della Tolleranza ai Guasti (Kill Worker) ---")
        
        def kill_worker_target():
            
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker...")
            # Logica per simulare il crash di un worker. 
            # Esempio: chiudere forzatamente una delle connessioni RPyC nell'orchestratore
            while not (hasattr(self.orchestrator, "connessioni_attive") and len(self.orchestrator.connessioni_attive) > 0):
                time.sleep(ft_cfg["kill_worker_after_seconds"])
            
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker...")
            try:
                with self.orchestrator.connessioni_lock:
                    if self.orchestrator.connessioni_attive:
                        # Prendiamo in modo sicuro la prima connessione aperta
                        target_conn = self.orchestrator.connessioni_attive[0]
                        target_conn.close()
                        print("[TEST TRIGGER] Connessione RPyC interrotta con successo!")
                    else:
                        print("[TEST TRIGGER WARNING] Nessuna connessione attiva rimasta al momento del kill.")
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
            "scenario_description": "Verifica della tolleranza ai guasti in caso di crash improvviso di un Worker RPC a metà computazione.",
            "status": status, 
            "kill_triggered_after_seconds": ft_cfg["kill_worker_after_seconds"], 
            "trees_built": num_trees, 
            "duration_seconds": duration 
        }