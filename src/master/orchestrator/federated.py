import pickle
import os
import random
import socket
import threading
import time
import rpyc
from rpyc.utils.classic import obtain
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig


class FederatedOrchestrator(BaseOrchestrator):
    
    def __init__(self, orchestrator_name: str = None):
        self.cfg = SystemConfig()
        name = orchestrator_name or f"Orchestrator-Federato-{socket.gethostname()}"
        super().__init__(
            orchestrator_name=name,
            queue_name="federated_queue"
        )
        self.current_job_id = None

    def _resolve_dataset_type(self, payload: dict) -> str:
        """Determina il tipo di dataset basandosi sul payload inviato dal Client."""
        dataset_type = payload.get("dataset_type")
        if dataset_type:
            return str(dataset_type).strip().lower()
        return "real"
    
    def _get_trees_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/trained_trees_{job_id}.pkl"
        return f"./.local_storage/checkpoints/trained_trees_{job_id}.pkl"

    def _get_inference_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/inference_chunks_{job_id}.pkl"
        return f"./.local_storage/inference_chunks_{job_id}.pkl"

    def _process_job(self, payload: dict, receipt_handle=None):
        """
        Punto di ingresso principale del Job Federato.
        Segue lo schema statico: interroga i worker associati agli shard,
        raccoglie gli alberi disponibili e applica lo scarto uniforme sul pool ottenuto.
        """
        job_id = payload.get("job_id")
        self.current_job_id = job_id
        hyperparameters = payload.get("hyperparameters", {})
        n_estimators = hyperparameters.get("n_estimators", 100)
        base_seed = hyperparameters.get("random_state", 42)

        print(f"\n[{self.orchestrator_name}] >>> Preso in carico Job Federato {job_id[:8]} <<<")
        
        try:
            # 1. Fase di addestramento parallelo (senza loop di riallocazione)
            totale_alberi = self._execute_training_step(
                payload=payload,
                target_alberi=n_estimators,
                seed=base_seed
            )
            
            # Ricarica la foresta finale (eventualmente campionata) per la validazione globale
            checkpoint_trees_path = self._get_trees_checkpoint_path(job_id)
            with open(checkpoint_trees_path, "rb") as f:
                foresta_globale = pickle.load(f)

            # 2. Validazione distribuita fault-tolerant sui soli nodi superstiti
            report_metriche = self._validate_global_forest(payload, foresta_globale)

            # 3. Salvataggio Report finale
            if report_metriche:
                report_path = f"./.local_storage/reports/report_federated_{job_id}.json"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, "w", encoding="utf-8") as rf:
                    import json
                    json.dump(report_metriche, rf, indent=4)
                print(f"[{self.orchestrator_name}] [SUCCESS] Report federato archiviato in {report_path}")

            if hasattr(self, 'state_manager') and self.state_manager:
                self.state_manager.update_request_status(job_id, "SUCCESSFUL", self.orchestrator_name)
                
            self._clean_checkpoint(job_id)
            print(f"[{self.orchestrator_name}] Job {job_id[:8]} completato con successo.")

        except Exception as e:
            print(f"[{self.orchestrator_name}] [CRITICAL ERRORE JOB] Pipeline fallita: {e}")
            traceback.print_exc()
            if hasattr(self, 'state_manager') and self.state_manager:
                self.state_manager.update_request_status(job_id, "FAILED", self.orchestrator_name)
        finally:
            if receipt_handle and hasattr(self, 'sqs_queue') and self.sqs_queue:
                try:
                    self.sqs_queue.delete_message(ReceiptHandle=receipt_handle)
                except Exception as ex:
                    print(f"[{self.orchestrator_name}] Impossibile eliminare messaggio SQS: {ex}")

    def _execute_training_step(self, payload: dict, target_alberi: int, seed: int) -> int:
        """
        Invia la richiesta a ciascun worker attivo per il proprio shard locale.
        Se un worker fallisce, viene registrato il dropout e si prosegue con i rimanenti.
        Alla fine, se gli alberi totali superano il target (over-provisioning), applica lo scarto uniforme.
        """
        job_id = payload.get("job_id")
        hyperparameters = payload.get("hyperparameters", {})
        dataset_type = self._resolve_dataset_type(payload)

        registry = ServiceRegistry()
        active_workers = list(registry.get_active_workers())

        if not active_workers:
            raise RuntimeError("Nessun worker attivo nel registro. Impossibile addestrare.")

        # Calcolo della quota base per worker (ipotizzando che tutti rispondano)
        quota_per_worker = max(1, (target_alberi + len(active_workers) - 1) // len(active_workers))
        print(f"[{self.orchestrator_name}] Allocazione iniziale: {quota_per_worker} alberi per ciascuno dei {len(active_workers)} worker.")

        collected_trees = []
        threads = []
        round_data = {}
        lock = threading.Lock()
        live_workers_at_validation = []

        def contact_worker(w_name, idx):
            try:
                w_info = registry.get_worker_info(w_name)
                if not w_info:
                    raise ConnectionError("Metadati worker mancanti nel registro")
                
                # Connessione RPC al worker proprietario dello shard
                conn = rpyc.connect(w_info["host"], w_info["port"], config={"allow_public_attrs": True})
                
                worker_payload = {
                    "job_id": job_id,
                    "dataset_type": dataset_type,
                    "n_estimators_local": quota_per_worker,
                    "worker_index": idx,
                    "hyperparameters": {
                        **hyperparameters, 
                        "random_state": seed + (idx * 1000)
                    }
                }
                
                raw_trees = conn.root.exposed_train_local_federated_forest(worker_payload)
                trained_list = obtain(raw_trees)
                conn.close()

                with lock:
                    round_data[w_name] = trained_list
                    live_workers_at_validation.append(w_name)
                    print(f"[{self.orchestrator_name}] -> Worker '{w_name}' ha risposto con {len(trained_list)} alberi.")
            except (socket.error, EOFError, Exception) as ex:
                # Se il worker muore, non c'è riassegnazione: i suoi dati sono locali e inaccessibili.
                print(f"[{self.orchestrator_name}] [CLIENT DROPOUT] Worker '{w_name}' non raggiungibile. Escluso dall'aggregazione: {ex}")

        # Esecuzione parallela
        for i, worker_name in enumerate(active_workers):
            t = threading.Thread(target=contact_worker, args=(worker_name, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Raccolta degli alberi inviati dai nodi superstiti
        for w_name, trees in round_data.items():
            if trees:
                collected_trees.extend(trees)

        if not collected_trees:
            raise RuntimeError("Tutti i nodi interessati sono falliti. Nessun albero raccolto per questo Job.")

        # APPLICAZIONE SCARTO UNIFORME FINALE
        # Se i nodi rimasti hanno prodotto una foresta più grande (o se vogliamo troncare al target dell'utente)
        if len(collected_trees) > target_alberi:
            print(f"[{self.orchestrator_name}] [SCARTO UNIFORME] Trovati {len(collected_trees)} alberi. Riduzione casuale a quota {target_alberi}.")
            collected_trees = random.sample(collected_trees, target_alberi)
        else:
            print(f"[{self.orchestrator_name}] Raccolti in totale {len(collected_trees)} alberi dai worker superstiti.")

        # Salvataggio del modello globale sul file system locale o S3
        checkpoint_trees_path = self._get_trees_checkpoint_path(job_id)
        os.makedirs(os.path.dirname(checkpoint_trees_path), exist_ok=True)
        with open(checkpoint_trees_path, "wb") as f:
            pickle.dump(collected_trees, f)
        
        tree_type = hyperparameters.get("tree_type", "classifier")
        final_count = self._reconstruct_and_save_global_model(collected_trees, tree_type)
        
        # Salvataggio checkpoint logico finale
        self._save_checkpoint(job_id, final_count, payload.get("retries", 0), seed)
        return final_count

    def _validate_global_forest(self, payload: dict, trained_trees: list) -> dict:
        """
        Invia la foresta globale finale ai soli nodi che hanno partecipato o sono online
        per validare il modello complessivo ciascuno sul proprio set di test locale.
        """
        print(f"\n[{self.orchestrator_name}] == AVVIO VALIDAZIONE FEDERATA DISTRIBUITA ==")
        job_id = payload.get("job_id")
        forest_bytes = pickle.dumps(trained_trees)
        
        registry = ServiceRegistry()
        active_worker_names = registry.get_active_workers()
        
        y_pred_global = []
        y_true_global = []
        results_lock = threading.Lock()
        threads = []

        def validate_worker(w_name):
            try:
                w_info = registry.get_worker_info(w_name)
                if not w_info:
                    return
                
                conn = rpyc.connect(w_info["host"], w_info["port"], config={"allow_public_attrs": True})
                raw_response = conn.root.exposed_predict_subset_forest(forest_bytes)
                worker_data = pickle.loads(obtain(raw_response))
                conn.close()

                with results_lock:
                    y_pred_global.extend(worker_data["y_pred"])
                    y_true_global.extend(worker_data["y_true"])
                    print(f"[{self.orchestrator_name}] Validazione completata su '{w_name}' ({worker_data['n_samples']} record).")
            except (socket.error, EOFError, Exception) as ex:
                print(f"[{self.orchestrator_name}] [VALIDATION DROPOUT] Impossibile validare su '{w_name}', saltato: {ex}")

        for worker_name in active_worker_names:
            t = threading.Thread(target=validate_worker, args=(worker_name,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if not y_pred_global:
            print(f"[{self.orchestrator_name}] [ERRORE] Nessun worker attivo ha risposto alla validazione finale.")
            return {}

        y_pred_arr = np.array(y_pred_global, dtype=np.int64)
        y_true_arr = np.array(y_true_global, dtype=np.int64)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        rep = classification_report(y_true_arr, y_pred_arr, zero_division=0)

        print(f"\n[{self.orchestrator_name}] === VALUTAZIONE AGGREGATA FEDERATA ===")
        print(f" • Accuracy:  {acc:.4f}")
        print(f" • Precision: {prec:.4f}")
        print(f" • Recall:    {rec:.4f}")
        print(f" • F1-Score:  {f1:.4f}")

        return {
            "job_id": job_id,
            "orchestrator": self.orchestrator_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "classification_report": rep,
            "timestamp": time.time()
        }

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int):
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

    def _clean_checkpoint(self, job_id: str):
        super()._clean_checkpoint(job_id)
        for path in [
            self._get_trees_checkpoint_path(job_id),
            self._get_inference_checkpoint_path(job_id)
        ]:
            if self.environment == "local" and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[{self.orchestrator_name}] [CLEAN OK] Rimosso file temporaneo: {path}")
                except Exception as e:
                    print(f"[{self.orchestrator_name}] [CLEAN WARN] Impossibile rimuovere {path}: {e}")
    
    def _reconstruct_and_save_global_model(self, all_trained_trees: list, tree_type: str) -> int:
        """
        Aggrega gli alberi raccolti in un oggetto RandomForest di Scikit-Learn
        e lo serializza su disco.
        """
        if not all_trained_trees:
            print(f"[{self.orchestrator_name}] Nessun albero collezionato.")
            return 0

        print(f"[{self.orchestrator_name}] Ricomposizione foresta globale conforme a Scikit-Learn...")
        try:
            # Assumiamo che gli alberi abbiano tutti lo stesso numero di feature di input
            n_features = all_trained_trees[0].n_features_in_
            
            if tree_type == "classifier":
                global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                # Nota: assunzione di classificazione binaria standard come da tuo snippet
                global_model.classes_ = np.array([0, 1], dtype=np.int64) 
                global_model.n_classes_ = 2
            else:
                global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
            
            # Iniezione manuale degli stimatori nell'oggetto sklearn
            global_model.estimators_ = all_trained_trees
            global_model.n_features_in_ = n_features
            global_model.n_outputs_ = 1
            
            # Salvataggio su disco
            TARGET_DIR = "./saved_models"
            os.makedirs(TARGET_DIR, exist_ok=True)
            model_path = os.path.join(TARGET_DIR, f"fed_model_{self.current_job_id}.pkl")
            
            with open(model_path, "wb") as f:
                pickle.dump(global_model, f)
            
            print(f"[{self.orchestrator_name}] Modello Globale salvato con successo in '{model_path}'.")
            return len(all_trained_trees)
            
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE AGGREGAZIONE] Fallimento durante l'unione dei sotto-modelli: {e}")
            traceback.print_exc()
            return len(all_trained_trees)


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()