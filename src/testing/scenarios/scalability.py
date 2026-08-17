from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig
from src.testing.scenarios.base import BaseTestScenario
import time
import os
import random

class ScalabilityScenario(BaseTestScenario):
    """Copre lo Scenario 2: Analisi della Scalabilità e del Throughput al variare dei Worker."""

    def run(self) -> dict:
        print("\n--- [SCENARIO 2] Test di Scalabilità e Throughput ---")
      
        scal_cfg = self.config.get("scalability_test", {})
        results = {}
        task_type = self.config.get("selected_task", "classifier")
        env = self.orchestrator.environment  
        all_active_workers = ServiceRegistry.get_available_workers(env)
        total_available = len(all_active_workers)
        workers_to_test = [w for w in scal_cfg.get("worker_counts_to_test", []) if w <= total_available]
        raw_metrics = {}

        if task_type == "classifier":
            target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
        else:
            target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)
        rng = random.Random(123)
        worker_ids = list(all_active_workers.keys())
        for worker_count in workers_to_test:
            print(f"[SCALABILITY LOCAL] Test con {worker_count} Worker attivi (Mock ServiceRegistry)...")
            sampled_ids = rng.sample(worker_ids, worker_count)
            sampled_workers = {k: all_active_workers[k] for k in sampled_ids}
            original_get_workers = ServiceRegistry.get_available_workers
            ServiceRegistry.get_available_workers = lambda environment: sampled_workers
            payload = self._build_payload(worker_count)
            try:
                # TIMING ADDESTRAMENTO
                start_train = time.perf_counter()
                self._reuse_dataset_if_available(payload, seed=123)
                num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
                train_duration = time.perf_counter() - start_train
                self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)
                
                # TIMING INFERENZA
                start_infer = time.perf_counter()
                accuracy_metrics = self._run_inference_and_get_metrics(payload, task_type)
                infer_duration = time.perf_counter() - start_infer
                
                # Calcolo throughput immediati
                train_throughput = num_trees / train_duration if train_duration > 0 else 0
                num_samples = accuracy_metrics.get("testing_set_size", 0)
                infer_throughput = num_samples / infer_duration if infer_duration > 0 else 0

                raw_metrics[worker_count] = {
                    "train_duration": train_duration,
                    "train_throughput": train_throughput,
                    "infer_duration": infer_duration,
                    "infer_throughput": infer_throughput,
                    "num_trees": num_trees,
                    "num_samples": num_samples,
                    "accuracy": accuracy_metrics
                }
            finally:
                ServiceRegistry.get_available_workers = original_get_workers
        # ─── FASE 2: CALCOLO SPEEDUP E STAMPA IN MODO ELEGANTE ───
        baseline_w = min(workers_to_test)
        base_train_time = raw_metrics[baseline_w]["train_duration"]
        base_infer_time = raw_metrics[baseline_w]["infer_duration"]
        
        print("\n" + "="*80)
        print(f"   REPORT DI SCALABILITÀ COMPLETO (Baseline di riferimento: {baseline_w} Worker)")
        print("="*80)

        for worker_count in workers_to_test:
            m = raw_metrics[worker_count]
            
            # Formula dello Speedup: Tempo con 1 Worker (o baseline) / Tempo con N Worker
            train_speedup = base_train_time / m["train_duration"] if m["train_duration"] > 0 else 1.0
            infer_speedup = base_infer_time / m["infer_duration"] if m["infer_duration"] > 0 else 1.0
            
            # Stampa a schermo strutturata
            print(f"\n[Configurazione: {worker_count} Worker]")
            print(f"    ADDESTRAMENTO ({m['num_trees']} alberi complessivi):")
            print(f"     • Durata:     {m['train_duration']:.2f} secondi")
            print(f"     • Throughput: {m['train_throughput']:.2f} alberi/s")
            print(f"     • Speedup:    {train_speedup:.2f}x")
            
            print(f"   INFERENZA :")
            print(f"     • Durata:     {m['infer_duration']:.2f} secondi")
            print(f"     • Speedup:    {infer_speedup:.2f}x")
            if task_type == "classifier":
                print(f"     • Metric:     Accuracy = {m['accuracy'].get('accuracy', 0.0)*100:.2f}%")
            else:
                print(f"     • Metric:     MSE = {m['accuracy'].get('mean_squared_error', 0.0):.4f}")
            
            # Salvataggio nel dizionario di output finale richiesto dall'orchestratore
            results[f"workers_{worker_count}"] = {
                "training": {
                    "duration_seconds": round(m["train_duration"], 2),
                    "throughput_trees_per_s": round(m['train_throughput'], 2),
                    "speedup": round(train_speedup, 2)
                },
                "inference": {
                    "duration_seconds": round(m["infer_duration"], 2),
                    "throughput_samples_per_s": round(m['infer_throughput'], 2),
                    "speedup": round(infer_speedup, 2)
                },
                "accuracy_metrics": m["accuracy"]
            }

        print("\n" + "="*80)

        return {
            "scenario_description": (
                "Strong scaling test completato per Addestramento ed Inferenza. "
                f"Carico fisso di {target_trees} alberi per ciascuna configurazione di worker testata."
            ),
            "execution_mode": "local",
            "scaling_type": "strong",
            "baseline_worker_count": baseline_w,
            "metrics_per_scale": results
        }

    def _build_payload(self, worker_count):
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        return {
            "job_id": f"test_scal_{worker_count}_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": hp,
        }
    
    def _run_inference_and_get_metrics(self, payload, task_type):
        """
        Esegue l'inferenza nativa dell'orchestratore e legge le metriche reali
        dal suo valore di ritorno (sia centralized.py che federated.py restituiscono
        {"metrics": {...}, "testing_set_size": ..., ...} da _execute_inference_step).
        Il modello è già salvato dal training precedente esattamente al path atteso
        da _resolve_model_path (./saved_models/model_{job_id}.pkl in entrambe le
        modalità): non serve nessun link/alias temporaneo.
        """
        accuracy_metrics = {}
        try:
            result = self.orchestrator._execute_inference_step(payload) or {}
            accuracy_metrics = dict(result.get("metrics", {}))
            accuracy_metrics["testing_set_size"] = result.get("testing_set_size", 0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR PERF TEST] Errore durante l'esecuzione dell'inferenza distribuita: {e}")

        # Fallback descrittivo in caso di fallimento dell'inferenza
        if not accuracy_metrics:
            print("[WARN PERF TEST] Impossibile estrarre metriche reali dall'inferenza. Verificare i log dei Worker.")
            if task_type == "classifier":
                accuracy_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                accuracy_metrics = {"mean_squared_error": 0.0}

        return accuracy_metrics