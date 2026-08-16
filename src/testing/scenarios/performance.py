import os
import time
from src.testing.scenarios.base import BaseTestScenario
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
class PerformanceAndMetricsScenario(BaseTestScenario):

    """Copre lo Scenario 1: Valutazione Prestazioni (Classif./Regr.) e Analisi Metriche."""
    
    def run(self) -> dict:
        print("[PERFORMANCE] Esecuzione in modalità LOCALE...")
        task_type = self.config.get("selected_task", "classifier")
        payload = self._build_payload()
        if task_type == "classifier":
            target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
        else:
            target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)
        
        start_time = time.perf_counter()
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
        duration = time.perf_counter() - start_time
        self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)
        
        throughput = num_trees / duration if duration > 0 else 0
        accuracy_metrics = self._run_inference_and_get_metrics(payload, task_type)

        return {
            "scenario_description": "Valutazione delle prestazioni pure di addestramento in esecuzione locale.",
            "status": "SUCCESS" if num_trees == target_trees else "FAILED",
            "execution_mode": "local",
            "duration_seconds": round(duration, 4),
            "trees_built": num_trees,
            "throughput_trees_per_sec": round(throughput, 4),
            "model_accuracy_metrics": accuracy_metrics
        }
    def _build_payload(self):
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        return {
            "job_id": f"test_perf_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", "synthetic/synthetic_dataset.csv"),
            "hyperparameters": hp
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