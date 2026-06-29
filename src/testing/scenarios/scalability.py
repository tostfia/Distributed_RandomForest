from testing.scenarios import BaseTestScenario
import time

class ScalabilityScenario(BaseTestScenario):
    """Copre lo Scenario 2: Analisi della Scalabilità e del Throughput al variare dei Worker."""
    def run(self) -> dict:
        print("\n--- [SCENARIO 2] Test di Scalabilità e Throughput ---")
        scal_cfg = self.config["scalability_test"]
        results = {}
        
        # Eseguiamo il test ciclicamente per i diversi numeri di worker configurati
        for worker_count in scal_cfg["worker_counts_to_test"]:
            print(f"Simulazione/Test con {worker_count} Worker attivi...")
            
            # Nota: In base a come gestisci i worker, qui potresti dover fare il prune 
            # temporaneo del ServiceRegistry o filtrare i canali RPC attivi nell'orchestrator.
            
            payload = {
                "job_id": f"test_scal_{worker_count}_{int(time.time())}",
                "dataset_type": self.config["dataset_type"],
                "dataset_path": self.config["dataset_path"],
                "hyperparameters": {
                    "n_estimators": scal_cfg["n_estimators_per_worker"] * worker_count,
                    "max_depth": 5,
                    "tree_type": self.config["selected_task"]
                }
            }
            
            start_time = time.perf_counter()
            total_target = scal_cfg["n_estimators_per_worker"] * worker_count
            num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=total_target, seed=42)
            duration = time.perf_counter() - start_time
            
            throughput = num_trees / duration if duration > 0 else 0
            results[f"worker_count_{worker_count}"] = {
                "duration": duration,
                "throughput": throughput,
                "trees": num_trees
            }
            print(f"-> Worker: {worker_count} | Tempo: {duration:.2f}s | Throughput: {throughput:.2f} alberi/s")
            
        return results