import os
import time
from src.testing.scenarios.base import BaseTestScenario
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
class PerformanceAndMetricsScenario(BaseTestScenario):
    """Copre lo Scenario 1: Valutazione Prestazioni (Classif./Regr.) e Analisi Metriche."""
    def run(self) -> dict:
        print(f"\n--- [SCENARIO 1 ] Test Prestazioni e Metriche per: {self.config['selected_task']} ---")
        task_type = self.config["selected_task"]
        target_trees = self.config["target_trees"]
        job_id = f"test_perf_{int(time.time())}"
        payload = {
            "job_id": job_id,
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {
                "n_estimators": target_trees,
                "max_depth": 5,
                "tree_type": self.config["selected_task"]
            }
        }
        
        start_time = time.perf_counter()
        # Invoca la logica dell'orchestrator passato come dipendenza
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = num_trees / duration if duration > 0 else 0

        accuracy_metrics = {}
        def intercept_metrics_centralized(predictions_matrix, y_test, tree_type, **kwargs):
            nonlocal accuracy_metrics
            if tree_type == "classifier":
                # Ricostruzione votazione speculare al codice dell'orchestrator centralizzato
                from sklearn.utils.extmath import weighted_mode
                uniform_weights = np.ones_like(predictions_matrix)
                final_predictions, _ = weighted_mode(predictions_matrix, uniform_weights, axis=0)
                final_predictions = final_predictions.ravel().astype(int)
                y_test = y_test.astype(int)
                
                accuracy_metrics = {
                    "accuracy": float(np.mean(final_predictions == y_test)),
                    "f1_score": float(f1_score(y_test, final_predictions, zero_division=0)),
                    "precision": float(precision_score(y_test, final_predictions, zero_division=0)),
                    "recall": float(recall_score(y_test, final_predictions, zero_division=0))
                }
            else:
                final_predictions = np.mean(predictions_matrix, axis=0)
                accuracy_metrics = {"mean_squared_error": float(np.mean((final_predictions - y_test) ** 2))}
        def intercept_metrics_federated(y_pred, y_true, tree_type, **kwargs):
            nonlocal accuracy_metrics
            if tree_type == "classifier":
                if np.issubdtype(y_pred.dtype, np.floating):
                    final_predictions = (y_pred >= 0.5).astype(int)
                else:
                    final_predictions = y_pred.astype(int)
                    
                y_true = y_true.astype(int)
                accuracy_metrics = {
                    "accuracy": float(np.mean(final_predictions == y_true)),
                    "f1_score": float(f1_score(y_true, final_predictions, zero_division=0)),
                    "precision": float(precision_score(y_true, final_predictions, zero_division=0)),
                    "recall": float(recall_score(y_true, final_predictions, zero_division=0))
                }
            else:
                accuracy_metrics = {"mean_squared_error": float(np.mean((y_pred.astype(float) - y_true.astype(float)) ** 2))}

        # Sostituzione temporanea (Monkey Patching sicuro per la durata del test)
        orig_centralized = getattr(self.orchestrator, "_print_and_validate_metrics", None)
        orig_federated = getattr(self.orchestrator, "_print_and_validate_metrics_federated", None)
        
        if orig_centralized:
            self.orchestrator._print_and_validate_metrics = intercept_metrics_centralized
        if orig_federated:
            self.orchestrator._print_and_validate_metrics_federated = intercept_metrics_federated

        modello_creato = os.path.join("./saved_models", f"cen_model_{job_id}.pkl")
        if not os.path.exists(modello_creato): # Prova anche la variante federata se applicabile
            modello_creato = os.path.join("./saved_models", f"fed_model_{job_id}.pkl")  
        modello_atteso_da_inferenza = os.path.join("./saved_models", f"model_{job_id}.pkl") 
        creato_link_temporaneo = False
        if os.path.exists(modello_creato) and not os.path.exists(modello_atteso_da_inferenza):
            try:
                # Creiamo un alias temporaneo così _execute_inference_step trova il file dove si aspetta di trovarlo
                os.link(modello_creato, modello_atteso_da_inferenza)
                creato_link_temporaneo = True
            except Exception as link_err:
                print(f"[WARN PERF TEST] Impossibile creare link temporaneo per il modello: {link_err}")
        try:
            # Eseguiamo l'inferenza nativa dell'orchestratore, che caricherà i dati di test reali estratti dal DAO
            self.orchestrator._execute_inference_step(payload)
        except Exception as e:
            print(f"[ERROR PERF TEST] Errore durante l'esecuzione dell'inferenza distribuita: {e}")
        finally:
            # Ripristino immediato dei metodi originali per non intaccare lo stato dell'orchestratore
            if orig_centralized: self.orchestrator._print_and_validate_metrics = orig_centralized
            if orig_federated: self.orchestrator._print_and_validate_metrics_federated = orig_federated
            if creato_link_temporaneo and os.path.exists(modello_atteso_da_inferenza):
                try:
                    os.remove(modello_atteso_da_inferenza)
                except:
                    pass

        # Fallback descrittivo in caso di fallimento catastrofico dell'inferenza
        if not accuracy_metrics:
            print("[WARN PERF TEST] Impossibile estrarre metriche reali dall'inferenza. Verificare i log dei Worker.")
            if task_type == "classifier":
                accuracy_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                accuracy_metrics = {"mean_squared_error": 0.0}

        return {
            "scenario_description": "Valutazione delle prestazioni pure di addestramento su un singolo nodo orchestratore.",
            "status": "SUCCESS" if num_trees == target_trees else "FAILED",
            "hyperparameters_tested": task_type,
            "duration_seconds": round(duration, 4),
            "trees_built": num_trees,
            "throughput_trees_per_sec": round(throughput, 4),
            "model_accuracy_metrics": accuracy_metrics
        }