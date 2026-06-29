from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig
from src.testing.scenarios.base import BaseTestScenario
import time

class ScalabilityScenario(BaseTestScenario):
    """Copre lo Scenario 2: Analisi della Scalabilità e del Throughput al variare dei Worker."""
    def run(self) -> dict:
        print("\n--- [SCENARIO 2] Test di Scalabilità e Throughput ---")
        scal_cfg = self.config["scalability_test"]
        results = {}
        
        env = self.orchestrator.environment  
        all_active_workers = ServiceRegistry.get_available_workers(env)
        total_available = len(all_active_workers)

        print(f"Worker totali rilevati nel ServiceRegistry per l'ambiente '{env}': {total_available}")
        
        workers_to_test = [
            w for w in scal_cfg["worker_counts_to_test"] 
            if w <= total_available
        ]
        
        print(f"Configurazione pianificata: {scal_cfg['worker_counts_to_test']}")
        print(f"Configurazione effettiva eseguibile: {workers_to_test}")
        # Eseguiamo il test ciclicamente per i diversi numeri di worker configurati
        for worker_count in workers_to_test:
            
            print(f"Simulazione/Test con {worker_count} Worker attivi...")
            
            sampled_workers = {k: all_active_workers[k] for k in list(all_active_workers.keys())[:worker_count]}

            original_get_workers = ServiceRegistry.get_available_workers
            ServiceRegistry.get_available_workers = lambda environment: sampled_workers
           
            
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
            try:
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
            
            finally:
                # 4. Ripristiniamo SEMPRE il comportamento originale del ServiceRegistry per non rompere i test successivi
                ServiceRegistry.get_available_workers = original_get_workers
            
           
            
        return results