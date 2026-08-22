import os
import time
from src.testing.scenarios.base import BaseTestScenario

class PerformanceAndMetricsScenario(BaseTestScenario):

    """Copre lo Scenario 1: Valutazione Prestazioni (Classif./Regr.) e Analisi Metriche."""

    def run(self) -> dict:
        execution_mode = getattr(self.orchestrator, "environment", "local")
        print(f"[PERFORMANCE] Esecuzione in ambiente '{execution_mode.upper()}'...")
        task_type = self.config.get("selected_task", "classifier")
        payload = self._build_payload()
        # Ricavato dal payload stesso (che ora nasce dal manifesto della
        # baseline): leggerlo separatamente da test_config.json permetteva di
        # chiedere N alberi mentre il payload ne dichiarava M.
        target_trees = self._resolve_target_trees()

        start_time = time.perf_counter()
        self._reuse_dataset_if_available(payload, seed=123)
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
        duration = time.perf_counter() - start_time
        self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)

        # Scomposizione del tempo, letta dall'orchestratore (vedi
        # CentralizedOrchestrator._execute_training_step). Serve a confrontare
        # con la baseline locale SOLO ciò che la baseline effettivamente fa:
        # 'training_only_seconds' è la costruzione degli alberi, il termine da
        # mettere accanto a T_seq/T_1node. ETL, aggregazione e stima OOB sono
        # costi propri dell'architettura distribuita, che la baseline non
        # sostiene affatto: vanno riportati, ma separati.
        timing = {
            "etl_seconds": round(getattr(self.orchestrator, "last_etl_seconds", 0.0), 4),
            "training_only_seconds": round(getattr(self.orchestrator, "last_dispatch_seconds", 0.0), 4),
            "aggregation_seconds": round(getattr(self.orchestrator, "last_aggregation_seconds", 0.0), 4),
            "oob_estimation_seconds": round(getattr(self.orchestrator, "last_oob_seconds", 0.0), 4),
            "_NOTA": "training_only_seconds e' il termine confrontabile con la baseline "
                     "(T_seq monocore / T_1node multicore). duration_seconds include anche "
                     "ETL, aggregazione e stima OOB, che la baseline non esegue.",
        }
        # Residuo non attribuito: differenza fra il totale misurato dallo
        # scenario e la somma delle fasi. Se cresce, significa che è comparso
        # del costo non ancora strumentato, invece di restare invisibile.
        timing["unaccounted_seconds"] = round(
            duration - sum(v for k, v in timing.items() if k.endswith("_seconds")), 4
        )

        throughput = num_trees / duration if duration > 0 else 0
        training_only = timing["training_only_seconds"]
        throughput_training_only = (num_trees / training_only) if training_only > 0 else 0
        accuracy_metrics = self._run_inference_and_get_metrics(payload, task_type)

        return {
            "scenario_description": f"Valutazione delle prestazioni pure di addestramento in esecuzione {execution_mode}.",
            "status": "SUCCESS" if num_trees == target_trees else "FAILED",
            "execution_mode": execution_mode,
            "duration_seconds": round(duration, 4),
            "timing_breakdown": timing,
            "trees_built": num_trees,
            "throughput_trees_per_sec": round(throughput, 4),
            "throughput_trees_per_sec_training_only": round(throughput_training_only, 4),
            "model_accuracy_metrics": accuracy_metrics
        }
    def _build_payload(self):
        # Iperparametri dal manifesto della baseline (vedi
        # BaseTestScenario._resolve_hyperparameters): è ciò che rende
        # confrontabili i tempi e le metriche di questo scenario con T_seq /
        # T_1node prodotti da run_baseline().
        hp = self._resolve_hyperparameters()
        payload = {
            "job_id": f"test_perf_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", "synthetic/synthetic_dataset.csv"),
            "hyperparameters": hp
        }
        if os.environ.get("SYS_MODE", "centralized") == "federated":
            payload = self._augment_payload_with_partitioning(payload)
        return payload

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