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
        task_type = self.config.get("selected_task", "classifier")
        env = self.orchestrator.environment  
        all_active_workers = ServiceRegistry.get_available_workers(env)
        total_available = len(all_active_workers)
        workers_to_test = [w for w in scal_cfg.get("worker_counts_to_test", []) if w <= total_available]
        raw_metrics = {}
        for worker_count in workers_to_test:
            print(f"[SCALABILITY LOCAL] Test con {worker_count} Worker attivi (Mock ServiceRegistry)...")
            sampled_workers = {k: all_active_workers[k] for k in list(all_active_workers.keys())[:worker_count]}
            original_get_workers = ServiceRegistry.get_available_workers
            ServiceRegistry.get_available_workers = lambda environment: sampled_workers
            total_target = scal_cfg.get("n_estimators_total", 60)
            payload = self._build_payload(worker_count, total_target)
            try:
                # TIMING ADDESTRAMENTO
                start_train = time.perf_counter() 
                num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=total_target, seed=123)
                train_duration = time.perf_counter() - start_train
                
                # TIMING INFERENZA
                start_infer = time.perf_counter()
                accuracy_metrics = self._mock_metrics_and_infer(payload, task_type)
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
        baseline_w = workers_to_test[0]
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
            print(f"     • Metric:     Accuracy = {m['accuracy'].get('accuracy', 0.0)*100:.2f}%")
            
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
                f"Carico totale fisso a ({scal_cfg.get('n_estimators_total', 60)} alberi)."
            ),
            "execution_mode": "local",
            "scaling_type": "strong",
            "baseline_worker_count": baseline_w,
            "metrics_per_scale": results
        }

    def _build_payload(self, worker_count,total_estimators):
        return {
            "job_id": f"test_scal_{worker_count}_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": {
                "n_estimators": total_estimators,
                "max_depth": 5,
                "tree_type": self.config.get("selected_task", "classifier")
            }
        }
    
    def _mock_metrics_and_infer(self, payload, task_type):
        # 1. Inizializziamo la variabile per permettere l'uso di nonlocal
        accuracy_metrics = {}
        
        # 2. Estraiamo il job_id dal payload perché ci serve per i percorsi dei file
        job_id = payload.get("job_id")

        def intercept_metrics_centralized(predictions_matrix, y_test, tree_type, **kwargs):
            nonlocal accuracy_metrics
            if tree_type == "classifier":
                # Ricostruzione votazione speculare al codice dell'orchestrator centralizzato
                from sklearn.utils.extmath import weighted_mode
                import numpy as np
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                uniform_weights = np.ones_like(predictions_matrix)
                final_predictions, _ = weighted_mode(predictions_matrix, uniform_weights, axis=0)
                final_predictions = final_predictions.ravel().astype(int)
                y_test = y_test.astype(int)
                n_classes = len(np.unique(np.concatenate([y_test, final_predictions])))
                avg_method = "binary" if n_classes <= 2 else "weighted"
                accuracy_metrics = {
                    "accuracy": float(np.mean(final_predictions == y_test)),
                    "f1_score": float(f1_score(y_test, final_predictions, average=avg_method, zero_division=0)),
                    "precision": float(precision_score(y_test, final_predictions, average=avg_method, zero_division=0)),
                    "recall": float(recall_score(y_test, final_predictions, average=avg_method, zero_division=0))
                }
            else:
                import numpy as np
                final_predictions = np.mean(predictions_matrix, axis=0)
                accuracy_metrics = {"mean_squared_error": float(np.mean((final_predictions - y_test) ** 2))}

        # 3. Rimosso "self" dai parametri, causa errore nel monkey patching su istanza
        def intercept_metrics_federated(y_pred, y_true, tree_type, **kwargs):
            nonlocal accuracy_metrics
            import numpy as np
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            if tree_type == "classifier":
                if np.issubdtype(y_pred.dtype, np.floating):
                    final_predictions = (y_pred >= 0.5).astype(int)
                else:
                    final_predictions = y_pred.astype(int)
                    
                y_true = y_true.astype(int)

                n_classes = len(np.unique(np.concatenate([y_true, final_predictions])))
                avg_method = "binary" if n_classes <= 2 else "weighted"
                accuracy_metrics = {
                    "accuracy": float(np.mean(final_predictions == y_true)),
                    "f1_score": float(f1_score(y_true, final_predictions, average=avg_method, zero_division=0)),
                    "precision": float(precision_score(y_true, final_predictions, average=avg_method, zero_division=0)),
                    "recall": float(recall_score(y_true, final_predictions, average=avg_method, zero_division=0))
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
                # Creiamo un alias temporaneo così _execute_inference_step trova il file
                os.link(modello_creato, modello_atteso_da_inferenza)
                creato_link_temporaneo = True
            except Exception as link_err:
                print(f"[WARN PERF TEST] Impossibile creare link temporaneo per il modello: {link_err}")
                
        try:
            # Eseguiamo l'inferenza nativa dell'orchestratore
            self.orchestrator._execute_inference_step(payload)
        except Exception as e:
            print(f"[ERROR PERF TEST] Errore durante l'esecuzione dell'inferenza distribuita: {e}")
        finally:
            # Ripristino immediato dei metodi originali
            if orig_centralized: self.orchestrator._print_and_validate_metrics = orig_centralized
            if orig_federated: self.orchestrator._print_and_validate_metrics_federated = orig_federated
            if creato_link_temporaneo and os.path.exists(modello_atteso_da_inferenza):
                try:
                    os.remove(modello_atteso_da_inferenza)
                except:
                    pass

        # Fallback descrittivo in caso di fallimento dell'inferenza
        if not accuracy_metrics:
            print("[WARN PERF TEST] Impossibile estrarre metriche reali dall'inferenza. Verificare i log dei Worker.")
            if task_type == "classifier":
                accuracy_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                accuracy_metrics = {"mean_squared_error": 0.0}

        # 4. Ritorniamo SOLO il dizionario delle metriche, come si aspetta _run_locally
        return accuracy_metrics
           