
import time
from src.testing.scenarios.base import BaseTestScenario




class PerformanceAndMetricsScenario(BaseTestScenario):
    """Copre lo Scenario 1 e 4: Valutazione Prestazioni (Classif./Regr.) e Analisi Metriche."""
    def run(self) -> dict:
        print(f"\n--- [SCENARIO 1 & 4] Test Prestazioni e Metriche per: {self.config['selected_task']} ---")
        
        payload = {
            "job_id": f"test_perf_{int(time.time())}",
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {
                "n_estimators": 30,
                "max_depth": 5,
                "tree_type": self.config["selected_task"]
            }
        }
        
        start_time = time.perf_counter()
        # Invoca la logica dell'orchestrator passato come dipendenza
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=30, seed=42)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = num_trees / duration if duration > 0 else 0
        
        # Qui il tuo sistema internamente chiamerà _print_and_validate_metrics()
        # Raccogliamo i risultati strutturati
        return {
            "status": "SUCCESS" if num_trees == 30 else "FAILED",
            "duration_seconds": duration,
            "trees_built": num_trees,
            "throughput_trees_per_sec": throughput
        }

