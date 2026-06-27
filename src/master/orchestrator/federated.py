import json
import pickle
import os
import random
import socket
import threading
import time
import queue
import rpyc
from rpyc.utils.classic import obtain
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

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
    
    

    
    def _execute_training_step(self, payload: dict,start_alberi: int, target_alberi: int, seed: int) -> int:
        """
        Invia la richiesta a ciascun worker attivo per il proprio shard locale.
        Se un worker fallisce, viene registrato il dropout e si prosegue con i rimanenti.
        Alla fine, se gli alberi totali superano il target (over-provisioning), applica lo scarto uniforme.
        """
        self.current_job_id = payload.get("job_id")
        
       
        if self.environment == "aws":
            checkpoint_trees_path = f"s3://my-cluster-datasets-bucket/checkpoints/checkpoint_trees_{self.current_job_id}.pkl"
        else:
            checkpoint_trees_path = f"./.local_storage/checkpoint_trees_{self.current_job_id}.pkl"
            os.makedirs("./.local_storage", exist_ok=True)

        all_trained_trees = []

        if start_alberi > 0:
            print(f"\n[{self.orchestrator_name}] [FAILOVER-RESUME] Rilevato start_alberi = {start_alberi}. Ripristino checkpoint fisico...")
            if os.path.exists(checkpoint_trees_path):
                try:
                    with open(checkpoint_trees_path, "rb") as f:
                        all_trained_trees = pickle.load(f)
                    print(f"[{self.orchestrator_name}] [OK] Ripristinati con successo {len(all_trained_trees)} alberi reali dal checkpoint.")
                    # Allineiamo lo start effettivo alla dimensione dell'array caricato per robustezza
                    start_alberi = len(all_trained_trees)
                except Exception as e_load:
                    print(f"[{self.orchestrator_name}] [ERROR] Checkpoint fisico corrotto: {e_load}. Ricalcolo da 0.")
                    start_alberi = 0
                    all_trained_trees = []
            else:
                print(f"[{self.orchestrator_name}] [WARN] File di checkpoint fisico non trovato a {checkpoint_trees_path}. Riparto da zero.")
                start_alberi = 0
        total_step_trees = target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare...")

        if total_step_trees <= 0:
            print(f"[{self.orchestrator_name}] Tutti gli alberi richiesti ({len(all_trained_trees)}) sono già pronti in memoria.")
        else:
            print(f"\n [{self.orchestrator_name}] Distribuzione carico residuo: {total_step_trees} alberi da generare...")
            while True:
                available_workers = ServiceRegistry.get_available_workers(self.environment)
                if available_workers:
                    print(f"[{self.orchestrator_name}] Worker rilevati: {list(available_workers.keys())}. Procedo...")
                    break
                
                print(f"[{self.orchestrator_name}] Nessun worker disponibile. In Attesa...")
                time.sleep(10)
            worker_names = list(available_workers.keys())
            num_workers = len(worker_names)

            hp = payload.get("hyperparameters", {})
            max_depth = hp.get("max_depth", None)
            tree_type = hp.get("tree_type", "classifier")

            CHUNK_SIZE = max(1, total_step_trees // (num_workers*2))  # Distribuzione iniziale più conservativa
            print(f"[{self.orchestrator_name}] Calcolo dinamico: {num_workers} worker rilevati -> CHUNK_SIZE impostata a {CHUNK_SIZE} alberi per task.")

            # 4. Configurazione della Coda di Sotto-Task locale
            task_queue = queue.Queue()
            sub_start = start_alberi
            task_id_counter = 1
            
            while sub_start < target_alberi:
                sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
                # Ogni sotto-task associa un seed specifico calcolato sull'offset cumulativo
                task_seed = seed + sub_start
                # Usiamo sub_start come offset assoluto rispetto al seed iniziale del JOB
                task_queue.put((task_id_counter, sub_start, sub_end, task_seed))
                task_id_counter += 1
                sub_start = sub_end
            feature_selezionate = []
            feature_selezionate = self.select_from_config()
            collected_trees = []
            
            
            results_lock = threading.Lock()
            connessioni_attive = []
            connessioni_lock = threading.Lock()
            active_worker_names = list(worker_names)

            def contact_worker(w_name, idx):
                w_info = available_workers[w_name]
                worker_conn = None
                try:
                    print(f" [RPC -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                    worker_conn = rpyc.connect(
                        w_info["host"], 
                        w_info["port"], 
                        config={
                            'allow_pickle': True,
                            'sync_request_timeout': 600,
                            'keepalive': True
                        }
                    )
                    with connessioni_lock:
                        connessioni_attive.append(worker_conn)

                    while len(all_trained_trees) < target_alberi:
                        try: 
                            task_id, start_t, end_t, chunk_seed = task_queue.get(timeout=2)
                        except queue.Empty:
                            with results_lock:
                                total_attuali = len(all_trained_trees)
                                num_worker_attivi = len(active_worker_names)
                            
                            if total_attuali >= target_alberi or num_worker_attivi <= 1:
                                break
                            time.sleep(1)
                            continue
                        quota_chunk = end_t - start_t
                        print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                        try:
                            result_raw = worker_conn.root.exposed_train_local_federated_forest(
                                job_id=self.current_job_id,
                                dataset_type=self._resolve_dataset_type(payload),
                                n_estimators_local=quota_chunk,
                                worker_index=idx,
                                hyperparameters={
                                    **hp, 
                                    "random_state": chunk_seed + (idx * 1000),
                                    "feature_selezionate": feature_selezionate
                                },
                            )
                            result_trees = pickle.loads(obtain(result_raw))
                            with results_lock:
                                all_trained_trees.extend(result_trees)
                                current_total = len(all_trained_trees)
                                # SALVATAGGIO FISICO ATOMICO PROGRESSIVO
                                try:
                                    with open(checkpoint_trees_path, "wb") as f_chk:
                                        pickle.dump(all_trained_trees, f_chk)
                                    print(f"   [RPC <- {w_name}] [CHECKPOINT FS OK] Task {task_id} archiviato. Progressivo in RAM/Storage: {current_total} alberi.")
                                except Exception as e_fs:
                                    print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere gli alberi parziali su file: {e_fs}")
                                
                                # Sincronizziamo in tempo reale anche il contatore logico nel Database/State Manager
                                if hasattr(self, 'state_manager') and self.state_manager:
                                    try:
                                        self.state_manager.update_request_status(
                                            job_id=self.current_job_id,
                                            status="PROCESSING",
                                            orchestrator_id=self.orchestrator_name,
                                            retries=payload.get("retries", 0),
                                            base_random_state=chunk_seed + (idx*1000),
                                            alberi_addestrati=current_total,
                                        )
                                    except Exception as e_db:
                                        print(f"   [ERRORE] Impossibile inviare l'heartbeat di stato a DynamoDB: {e_db}")
                                
                            print(f"   [RPC <- {w_name}] Task {task_id} completato. Ricevuti {len(result_trees)} alberi.")
                            task_queue.task_done()
                        except Exception as e:
                            print(f"   [ERRORE RPC] Fallimento o disconnessione del worker {w_name} durante il Task {task_id}: {e}")
                            
                            # FAULT TOLERANCE REALE: Reinserimento immediato del chunk per la fault tolerance
                            task_queue.put((task_id, start_t, end_t, chunk_seed))
                            print(f"[{self.orchestrator_name}-Thread] Task {task_id} riaccodato con successo per il failover.")
                            
                            with results_lock:
                                if w_name in active_worker_names:
                                    active_worker_names.remove(w_name)
                            break 
                except Exception as conn_err:
                    print(f"   [ERRORE CRITICO] Impossibile connettersi a {w_name}: {conn_err}")
                    with results_lock:
                        if w_name in active_worker_names:
                            active_worker_names.remove(w_name)
                finally:
                    if worker_conn:
                        try:
                            worker_conn.close()
                            with connessioni_lock:
                                if worker_conn in connessioni_attive:
                                    connessioni_attive.remove(worker_conn)
                        except:
                            pass
            threads = []
            # Esecuzione parallela
            for i, worker_name in enumerate(worker_names):
                t = threading.Thread(target=contact_worker, args=(worker_name, i))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            if not task_queue.empty() and len(active_worker_names) == 0:
                print(f"   [{self.orchestrator_name}] Tutti i worker sono crashati. SQS gestirà il failover macro.")
                raise RuntimeError("Sotto-sistema Fault Tolerance interrotto: Nessun worker disponibile rimasto.")

            
            if not all_trained_trees:
                raise RuntimeError("Tutti i nodi interessati sono falliti. Nessun albero raccolto per questo Job.")

            # APPLICAZIONE SCARTO UNIFORME FINALE
            if len(all_trained_trees) > target_alberi:
                print(f"[{self.orchestrator_name}] [SCARTO UNIFORME] Trovati {len(all_trained_trees)} alberi. Riduzione casuale a quota {target_alberi}.")
                collected_trees = random.sample(all_trained_trees, target_alberi)
            else:
                print(f"[{self.orchestrator_name}] Raccolti in totale {len(all_trained_trees)} alberi dai worker superstiti.")
                collected_trees = all_trained_trees

            # Salvataggio del modello globale sul file system locale o S3
            checkpoint_trees_path = self._get_trees_checkpoint_path(self.current_job_id)
            os.makedirs(os.path.dirname(checkpoint_trees_path), exist_ok=True)
            with open(checkpoint_trees_path, "wb") as f:
                pickle.dump(collected_trees, f)
            
            
            final_count = self._reconstruct_and_save_global_model(collected_trees, tree_type)
            
            # Salvataggio checkpoint logico finale
            self._save_checkpoint(self.current_job_id, final_count, payload.get("retries", 0), seed)
            return final_count

    def _execute_inference_step(self, payload: dict) -> dict:
    
        print(f"\n[{self.orchestrator_name}] == AVVIO VALIDAZIONE FEDERATA DISTRIBUITA ==")
        job_id = payload.get("job_id")
        hyperparameters = payload.get("hyperparameters", {})
        tree_type = hyperparameters.get("tree_type", "classifier")

        inference_start_time = time.perf_counter()

        # 1. RISOLUZIONE PERCORSO MODELLO
        if self.environment == "aws":
            model_path = f"s3://my-cluster-datasets-bucket/saved_models/fed_model_{job_id}.pkl"
        else:
            model_path = os.path.join("./saved_models", f"fed_model_{job_id}.pkl")

        # 2. CARICAMENTO DELLA FORESTA (MODELLO GLOBALE AGGREGATO)
        if self.environment == "local":
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")
            print(f"[{self.orchestrator_name}] Caricamento della foresta locale da {model_path}...")
            with open(model_path, "rb") as f:
                global_model = pickle.load(f)
        else:
            print(f"[{self.orchestrator_name}] Ambiente AWS: caricamento foresta...")
            local_fallback_path = os.path.join("./saved_models", f"fed_model_{job_id}.pkl")
            with open(local_fallback_path, "rb") as f:
                global_model = pickle.load(f)

        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta caricata. Numero totale di alberi: {total_trees}")

        # 3. SCOPERTA WORKER
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        if num_workers == 0:
            raise RuntimeError("Nessun worker disponibile per l'inferenza federata.")
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        # 4. SERIALIZZAZIONE UNICA DELLA FORESTA INTERA (inviata identica a ogni worker)
        forest_bytes = pickle.dumps(all_trees)
        feature_selezionate = self.select_from_config()

        # 5. STRUTTURE DATI CONDIVISE
        # Usiamo liste mutabili per evitare nonlocal su tipi immutabili
        y_pred_global = []
        y_true_global = []
        total_samples_ref = [0]   # [0] = contenitore mutabile, no nonlocal needed
        failed_workers = set()

        results_lock = threading.Lock()
        connessioni_attive = []
        connessioni_lock = threading.Lock()
        active_worker_names = list(worker_names)

        # 6. FUNZIONE CONSUMATRICE: una chiamata RPC per worker, foresta intera
        def validate_worker(w_name, idx):
            conn = None
            try:
                w_info = available_workers[w_name]
                if not w_info:
                    return

                print(f" [RPC INF -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                conn = rpyc.connect(
                    w_info["host"], w_info["port"],
                    config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": 300}
                )
                with connessioni_lock:
                    connessioni_attive.append(conn)

                print(f"[{self.orchestrator_name}-InfThread] Invio foresta completa ({total_trees} alberi) a {w_name}...")
                raw_response = conn.root.exposed_predict_subset_forest(payload=pickle.dumps({
                    "forest": forest_bytes,
                    "job_id": job_id,
                    "worker_index": idx,
                    "hyperparameters": {
                        **hyperparameters,
                        "dataset_type": self._resolve_dataset_type(payload),
                        "feature_selezionate": feature_selezionate,
                        "tree_type": tree_type,
                    }
                }))
                worker_data = pickle.loads(obtain(raw_response))

                with results_lock:
                    y_pred_global.extend(worker_data["y_pred"])
                    y_true_global.extend(worker_data["y_true"])
                    total_samples_ref[0] += worker_data["n_samples"]
                    print(f"[{self.orchestrator_name}] Validazione completata su '{w_name}' ({worker_data['n_samples']} record).")

            except Exception as ex:
                print(f"   [ERRORE INF] Fallimento su '{w_name}': {ex}")
                with results_lock:
                    if w_name in active_worker_names:
                        active_worker_names.remove(w_name)
                    failed_workers.add(w_name)
            finally:
                if conn:
                    try:
                        conn.close()
                        with connessioni_lock:
                            if conn in connessioni_attive:
                                connessioni_attive.remove(conn)
                    except:
                        pass

        # 7. AVVIO MULTI-THREADING (un thread per worker)
        rpc_start_time = time.perf_counter()
        threads = []
        for idx, name in enumerate(worker_names):
            t = threading.Thread(target=validate_worker, args=(name, idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Chiusura precauzionale connessioni residue
        with connessioni_lock:
            for conn in connessioni_attive:
                try: conn.close()
                except Exception: pass

        rpc_inference_time = time.perf_counter() - rpc_start_time

        # 8. GUARD: almeno un worker deve aver risposto
        if not y_pred_global:
            print(f"[{self.orchestrator_name}] [ERRORE] Nessun worker ha risposto alla validazione federata.")
            return {}

        if failed_workers:
            print(f"[{self.orchestrator_name}] [WARN] {len(failed_workers)} worker non hanno risposto: {failed_workers}. Metriche calcolate sui rimanenti.")

        total_inference_time = time.perf_counter() - inference_start_time

        # 9. CALCOLO E STAMPA METRICHE AGGREGATE
        self._print_and_validate_metrics_federated(
            y_pred=np.array(y_pred_global, dtype=np.int64),
            y_true=np.array(y_true_global, dtype=np.int64),
            tree_type=tree_type,
            testing_set_size=total_samples_ref[0],
            job_id=job_id,
            total_inference_time=total_inference_time,
            rpc_inference_time=rpc_inference_time
        )
            
   
    
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
        
    def _print_and_validate_metrics_federated(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        tree_type: str,
        testing_set_size: int,
        job_id: str,
        total_inference_time: float,
        rpc_inference_time: float
    ):
        

        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE PRESTAZIONI MODELLO FEDERATO (JOB: {job_id[:8]})")
        print("═" * 75)
        print(f"  TEMPO TOTALE DI INFERENZA:              {total_inference_time:.4f} secondi")
        print(f"  TEMPO INFERENZA DISTRIBUITA RPC:        {rpc_inference_time:.4f} secondi")
        print("═" * 75 + "\n")

        if tree_type == "classifier":
            final_predictions = y_pred.astype(int)
            y_true = y_true.astype(int)

            accuracy  = np.mean(final_predictions == y_true)
            precision = precision_score(y_true, final_predictions, zero_division=0)
            recall    = recall_score(y_true, final_predictions, zero_division=0)
            f1        = f1_score(y_true, final_predictions, zero_division=0)
            cm        = confusion_matrix(y_true, final_predictions)

            print(f"  Tipo di Modello:                        CLASSIFICATORE FEDERATO")
            print(f"  Testing Set size (aggregato):           {testing_set_size} campioni")
            print("-" * 75)
            print(f"  ACCURACY FINALE FEDERATA:               {accuracy * 100:.2f} %")
            print(f"  PRECISION FEDERATA:                     {precision * 100:.2f} %")
            print(f"  RECALL FEDERATA:                        {recall * 100:.2f} %")
            print(f"  F1-SCORE FEDERATO:                      {f1 * 100:.2f} %")
            print("-" * 75)
            print("  Matrice di Confusione:")
            print(cm)
            print("\n  Classification Report Completo:")
            print(classification_report(y_true, final_predictions, zero_division=0))
        else:
            final_predictions = y_pred.astype(float)
            mae = np.mean(np.abs(final_predictions - y_true.astype(float)))
            print(f"  Tipo di Modello:                        REGRESSORE FEDERATO")
            print(f"  Testing Set size (aggregato):           {testing_set_size} campioni")
            print(f"  MAE FINALE FEDERATO:                    {mae:.4f}")

        print("═" * 75 + "\n")
        
    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int, alberi_reali: list = None):
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

        if alberi_reali is not None and len(alberi_reali) > 0:
            if self.environment == "aws":
                checkpoint_trees_path = f"s3://my-cluster-datasets-bucket/checkpoints/checkpoint_trees_{job_id}.pkl"
            else:
                checkpoint_trees_path = f"./.local_storage/checkpoint_trees_{job_id}.pkl"
            try: 
                with open(checkpoint_trees_path, "wb") as f:
                    pickle.dump(alberi_reali, f)
                print(f"[{self.orchestrator_name}] Checkpoint alberi salvato in {checkpoint_trees_path}.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE CHECKPOINT] Impossibile salvare checkpoint alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        super()._clean_checkpoint(job_id)
        if self.environment == "aws":
            checkpoint_trees_path = f"s3://my-cluster-datasets-bucket/checkpoints/checkpoint_trees_{job_id}.pkl"
        else:
            checkpoint_trees_path = f"./.local_storage/checkpoint_trees_{job_id}.pkl"
        if os.path.exists(checkpoint_trees_path):
            try:
                os.remove(checkpoint_trees_path)
                print(f"[{self.orchestrator_name}] Checkpoint alberi rimosso da {checkpoint_trees_path}.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE CLEANUP] Impossibile rimuovere checkpoint alberi: {e}")
        inference_cp = self._get_inference_checkpoint_path(job_id)
        if os.path.exists(inference_cp):
            os.remove(inference_cp)
        
    def _get_trees_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/trained_trees_{job_id}.pkl"
        return f"./.local_storage/checkpoints/trained_trees_{job_id}.pkl"

    def _get_inference_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/inference_chunks_{job_id}.pkl"
        return f"./.local_storage/inference_chunks_{job_id}.pkl"
    
    def _load_inference_checkpoint(self, job_id: str):
        path = self._get_inference_checkpoint_path(job_id)
        if self.environment == "local" and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    chunks = pickle.load(f)
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Caricati {len(chunks)} chunk di inferenza dal checkpoint.")
                return chunks
            except Exception as e:
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Errore nel caricamento del checkpoint: {e}")
        return []
    
    def select_from_config(self):
        # Percorso 1: relativo al cwd (funziona se lanciato da root con python main.py)
        config_path = os.path.join(os.getcwd(), "outputs_baseline", "config.json")
        
        # Percorso 2: fallback risalendo dalla posizione del file sorgente
        if not os.path.exists(config_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
            config_path = os.path.join(project_root, "outputs_baseline", "config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_dati = json.load(f)
                feature_selezionate = config_dati.get("feature_selezionate", None)
                if not feature_selezionate:
                    print(f"[{self.orchestrator_name}] [ATTENZIONE] 'feature_selezionate' assente o vuoto nel config.")
                    return None
                print(f"[{self.orchestrator_name}] Config caricata da {config_path}. Trovate {len(feature_selezionate)} feature.")
                return feature_selezionate
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE] Lettura config fallita: {e}")
        else:
            print(f"[{self.orchestrator_name}] [ATTENZIONE] config.json non trovato in nessuno dei percorsi:")
            print(f"  • {config_path}")
        return None


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()