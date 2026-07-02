from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig
from src.testing.scenarios.base import BaseTestScenario
import time
import os


class ScalabilityScenario(BaseTestScenario):
    """Copre lo Scenario 2: Analisi della Scalabilità e del Throughput al variare dei Worker."""

   

    def run(self) -> dict:
        print("\n--- [SCENARIO 2] Test di Scalabilità e Throughput ---")
      
        scal_cfg = self.config.get("scalability_test", {})
        results = {}
        env = self.orchestrator.environment  
        all_active_workers = ServiceRegistry.get_available_workers(env)
        total_available = len(all_active_workers)
        workers_to_test = [w for w in scal_cfg.get("worker_counts_to_test", []) if w <= total_available]
        
        for worker_count in workers_to_test:
            print(f"[SCALABILITY LOCAL] Test con {worker_count} Worker attivi (Mock ServiceRegistry)...")
            sampled_workers = {k: all_active_workers[k] for k in list(all_active_workers.keys())[:worker_count]}
            original_get_workers = ServiceRegistry.get_available_workers
            ServiceRegistry.get_available_workers = lambda environment: sampled_workers
           
            payload = self._build_payload(worker_count, scal_cfg)
            try:
                start_time = time.perf_counter()
                total_target = scal_cfg.get("n_estimators_per_worker", 20) * worker_count
                num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=total_target, seed=42)
                duration = time.perf_counter() - start_time
                throughput = num_trees / duration if duration > 0 else 0
                results[f"workers_{worker_count}"] = {"throughput": round(throughput, 2)}
                print(f"-> Worker: {worker_count} | Tempo: {duration:.2f}s | Throughput: {throughput:.2f} alberi/s")
            finally:
                ServiceRegistry.get_available_workers = original_get_workers

        return {
            "scenario_description": "Analisi del throughput e della scalabilità locale.",
            "execution_mode": "local",
            "metrics_per_scale": results
        }

    def _build_payload(self, worker_count, scal_cfg):
        return {
            "job_id": f"test_scal_{worker_count}_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": {
                "n_estimators": scal_cfg.get("n_estimators_per_worker", 20) * worker_count,
                "max_depth": 5,
                "tree_type": self.config.get("selected_task", "classifier")
            }
        }
           