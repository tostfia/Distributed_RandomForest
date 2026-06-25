import pickle
import os
import queue
import socket
import threading
import time
import rpyc
from rpyc.utils.classic import obtain
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.federated_data_splitter import FederatedDataSplitter
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig


class FederatedOrchestrator(BaseOrchestrator):
    # In FederatedOrchestrator
    def __init__(self, orchestrator_name: str = None):
        self.cfg = SystemConfig()
        import socket
        name = orchestrator_name or f"Orchestrator-Federato-{socket.gethostname()}"
        super().__init__(
            orchestrator_name=name,
            queue_name="federated_queue"
        )
        self.current_job_id = None
        self.worker_shards_paths = {}  # Mappa per tracciare i path dei dataset partizionati per ciascun worker

    def _resolve_dataset_type(self, payload: dict) -> str:
        dataset_type = payload.get("dataset_type")
        if dataset_type: 
            return str(dataset_type).strip().lower()
        return "real"
    
    def _prepare_data(self, payload: dict, base_seed: int):
        # 1. Istanzio il loader corretto
        loader = RawCSVDataLoader(data_url="./dataset_cache") 
        
        # 2. Split e Sharding (Scrittura fisica su disco)
        splitter = FederatedDataSplitter(random_state=base_seed)
        splitter.split_and_shard(loader=loader, num_workers=len(self.workers))
        
        # 3. Broadcast di caricamento ai Worker via RPC
        for name, info in self.workers.items():
            try:
                conn = rpyc.connect(info["host"], info["port"])
                conn.root.exposed_load_local_shard()
                conn.close()
                print(f"[Orchestrator] Worker {name} ha caricato il suo shard.")
            except Exception as e:
                print(f"[Orchestrator] Errore nel caricamento del worker {name}: {e}")
    

  

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> bool:
        """
        Esegue un round di addestramento Federated Learning per Random Forest.
        I worker addestrano gli alberi localmente sui propri dati privati.
        L'orchestratore colleziona gli alberi e li aggrega nel modello globale.
        """

        # 1. PARAMETRIZZAZIONE FEDERATA (Niente ETL centralizzato)
        # Recuperiamo l'identificativo del dataset locale che i client devono usare 
        
        expected_job_id = payload.get("job_id", "unknown_job")
        self.current_job_id = expected_job_id
        dataset_tag = self._resolve_dataset_type(payload)
        try:
            self._prepare_data(payload, seed)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE CRITICO ENTRATA ETL] Pipeline interrotta: {e}")
            return False

        # Configurazione path di checkpoint (per ripartire in caso di crash dell'orchestratore)
        if self.environment == "aws":
            checkpoint_trees_path = f"s3://my-cluster-datasets-bucket/checkpoints/federated_trees_{self.current_job_id}.pkl"
        else:
            checkpoint_trees_path = f"./.local_storage/federated_trees_{self.current_job_id}.pkl"
            os.makedirs("./.local_storage", exist_ok=True)
        
        all_trained_trees = []

        # ─── FASE DI RESUME (FAULT TOLERANCE DELL'ORCHESTRATORE) ───
        if start_alberi > 0:
            print(f"\n[{self.orchestrator_name}] [FEDERATED-RESUME] Ripristino checkpoint globale...")
            if os.path.exists(checkpoint_trees_path):
                try:
                    with open(checkpoint_trees_path, "rb") as f:
                        all_trained_trees = pickle.load(f)
                    print(f"[{self.orchestrator_name}] [OK] Ripristinati {len(all_trained_trees)} alberi federati dal checkpoint.")
                    start_alberi = len(all_trained_trees)
                except Exception as e_load:
                    print(f"[{self.orchestrator_name}] [ERROR] Checkpoint corrotto: {e_load}. Ricalcolo round da 0.")
                    start_alberi = 0
                    all_trained_trees = []
            else:
                print(f"[{self.orchestrator_name}] [WARN] Checkpoint non trovato. Riparto da zero.")
                start_alberi = 0

        total_residual_trees = target_alberi - start_alberi
        if total_residual_trees <= 0:
            print(f"[{self.orchestrator_name}] Tutti gli alberi richiesti sono già pronti in memoria.")
        else:
            print(f"\n[{self.orchestrator_name}] Inizio Round Federato: {total_residual_trees} alberi residui da raccogliere...")

            # 2. SCOPERTA DEI NODI FEDERATI (WORKER)
            while True:
                available_workers = ServiceRegistry.get_available_workers(self.environment)
                if available_workers:
                    print(f"[{self.orchestrator_name}] Nodi Federati attivi rilevati: {list(available_workers.keys())}.")
                    break
                print(f"[{self.orchestrator_name}] Nessun nodo federato disponibile. In attesa di client...")
                time.sleep(10)

            worker_names = list(available_workers.keys())
            num_workers = len(worker_names)

            # Estrattori Iperparametri
            hp = payload.get("hyperparameters", {})
            max_depth = hp.get("max_depth", None)
            tree_type = hp.get("tree_type", "classifier")

            # 3. RIPARTIZIONE DEL CARICO TRA I NODI CLIENT
            # Nel FL puro, ogni nodo contribuisce all'addestramento proporzionalmente o equamente sui propri dati.
            # Calcoliamo quanti alberi deve generare OGNI singolo worker in questo round.
            CHUNK_SIZE = max(1, total_residual_trees // num_workers)
            print(f"[{self.orchestrator_name}] Configurazione Round: {CHUNK_SIZE} alberi richiesti a ogni Client.")

            threads = []
            for i, name in enumerate(worker_names):
                sub_start = start_alberi + (i * CHUNK_SIZE)
                sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
                
                # Gestione del resto matematico per l'ultimo worker
                if i == num_workers - 1:
                    sub_end = target_alberi
                    
                chunk_seed = seed + (sub_start - start_alberi)
                
                # Passiamo i range precisi come argomenti del thread per quel worker specifico
                t = threading.Thread(
                    target=federated_worker_consumer, 
                    args=(name, sub_start, sub_end, chunk_seed)
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            results_lock = threading.Lock()
            connessioni_attive = []
            connessioni_lock = threading.Lock()
            active_worker_names = list(worker_names)

            # 4. CONSUMER THREAD: COORDINAMENTO RPC FEDERATO
            def federated_worker_consumer(w_name):
                w_info = available_workers[w_name]
                worker_conn = None
                try:
                    print(f" [Federated RPC -> {w_name}] Connessione al nodo privato...")
                    worker_conn = rpyc.connect(
                        w_info["host"], 
                        w_info["port"], 
                        config={
                            'allow_pickle': True,
                            'sync_request_timeout': 600,  # Timeout alto per l'addestramento locale
                            'keepalive': True
                        }
                    )
                    with connessioni_lock:
                        connessioni_attive.append(worker_conn)
                    quota_chunk = end_t - start_t
                    print(f" [Federated RPC -> {w_name}] Addestramento assegnato di {quota_chunk} alberi (range {start_t}-{end_t}) con seed {chunk_seed}...")

                    # Chiamata RPC sul blocco assegnato
                    result_raw = worker_conn.root.exposed_train_local_federated_forest(
                        dataset_tag=dataset_tag, 
                        num_trees=quota_chunk,       
                        base_seed=chunk_seed,    
                        max_depth=max_depth
                    )
                    result_trees = pickle.loads(obtain(result_raw))
            
                    with results_lock:
                        all_trained_trees.extend(result_trees)
                        current_total = len(all_trained_trees)
                        
                        # Checkpoint progressivo dell'aggregazione globale
                        try:
                            with open(checkpoint_trees_path, "wb") as f_chk:
                                pickle.dump(all_trained_trees, f_chk)
                        except Exception as e_fs:
                            print(f" [ERRORE CHECKPOINT] Impossibile aggiornare file di round: {e_fs}")
                        
                        # Aggiornamento State Manager/Dashboard
                        if hasattr(self, 'state_manager') and self.state_manager:
                            try:
                                self.state_manager.update_request_status(
                                    job_id=self.current_job_id,
                                    status="PROCESSING",
                                    orchestrator_id=self.orchestrator_name,
                                    retries = payload.get("retries", 0),
                                    base_random_state = seed,
                                    alberi_addestrati=current_total
                                )
                            except Exception as e_sm:
                                print(f" [ERRORE] Impossibile aggiornare lo stato del job: {e_sm}")
                    

                    

                except Exception as e:
                    print(f" [FALLIMENTO NODO FEDERATO] Il client {w_name} è disconnesso o ha fallito: {e}")
                            
                    
                except Exception as e_outer:
                    print(f" [ERRORE GENERALE] Errore sul nodo {w_name}: {e_outer}")
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
                        except: pass

            # 5. AVVIO DELLA PARALIZZAZIONE DEI NODI
            # Lanciamo un pool di thread pari al numero di nodi federati pronti a lavorare
         
            for name in worker_names:
                t = threading.Thread(target=federated_worker_consumer, args = (name,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()


            # Pulizia Connessioni
            with connessioni_lock:
                for conn in connessioni_attive:
                    try: conn.close()
                    except Exception: pass

            # 7. AGGREGAZIONE FEDERATA (Server-Side Model Fusion)
            # Nelle Random Forest, l'aggregazione federata consiste nell'unire gli stimatori locali
            # per formare un'unica grande foresta globale (Federated Ensemble Integration).
            if len(all_trained_trees) > 0:
                print(f"[{self.orchestrator_name}] Inizio fusione dei modelli locali nella foresta globale...")
                try:
                    n_features = all_trained_trees[0].n_features_in_
                    
                    if tree_type == "classifier":
                        global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                        global_model.classes_ = np.array([0, 1])  # Standardizziamo le classi attese
                        global_model.n_classes_ = 2
                    else:
                        global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
                    
                    # Iniettiamo gli alberi estratti da tutti i client segregati
                    global_model.estimators_ = all_trained_trees
                    global_model.n_features_in_ = n_features
                    global_model.n_outputs_ = 1
                    
                    TARGET_DIR = "./saved_models"
                    os.makedirs(TARGET_DIR, exist_ok=True)
                    model_path = os.path.join(TARGET_DIR, f"fedetated_model_{self.current_job_id}.pkl")
                    
                    with open(model_path, "wb") as f:
                        pickle.dump(global_model, f)
                    
                    print(f"[{self.orchestrator_name}] Modello Globale Federato salvato in '{model_path}'.")
                    return True
                    
                except Exception as e:
                    print(f" [ERRORE FUSIONE FEDERATA] Impossibile accorpare i sotto-modelli dei client: {e}")
                    traceback.print_exc()
                    return False

            print(f"[{self.orchestrator_name}] Nessun modello parziale collezionato dai client.")
            return False
    
    def _execute_inference_step(self, payload: dict):
        job_id = payload.get("job_id", "unknown_job")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        print(f"[{self.orchestrator_name}] Avvio fase di inferenza federata per il job {job_id}...")
        inference_start_time = time.perf_counter()
        if self.environment == "aws":
            model_path = f"s3://my-cluster-datasets-bucket/models/federated_checkpoint_{job_id}.pkl"
        else:
            model_path = f"./.saved_models/federated_model_{job_id}.pkl"

        print(f"[{self.orchestrator_name}] Caricamento modello globale federato da '{model_path}'...")
        if self.environment == "local":
            if not os.path.exists(model_path):
                print(f"[{self.orchestrator_name}] Modello globale federato non trovato in locale. Abort inferenza.")
                return
            with open(model_path, "rb") as f:
                global_model = pickle.load(f)
        else:
            local_fallback_path = os.path.join("./.saved_models", f"federated_model_{job_id}.pkl")
            if not os.path.exists(local_fallback_path):
                print(f"[{self.orchestrator_name}] Modello globale federato non trovato in locale. Abort inferenza.")
                return
            with open(local_fallback_path, "rb") as f:
                global_model = pickle.load(f)
        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Modello globale federato caricato con {total_trees} alberi.")    
        
        serialized_global_forest = pickle.dumps(all_trees)
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            print(f"[{self.orchestrator_name}] Nessun nodo federato disponibile per l'inferenza. Abort.")
            return
        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Nodi federati attivi per inferenza: {worker_names}.")
        
        task_queue = queue.Queue()
        task_id_counter = 0
        inference_cp_path = self._get_inference_checkpoint_path(job_id)
        results_chunks = self._load_inference_checkpoint(job_id)
        already_done_workers = {w_name for w_name, _ in results_chunks}

        for w_name in worker_names:
            if w_name not in already_done_workers:
                task_queue.put((task_id_counter, w_name, serialized_global_forest))
                task_id_counter += 1
            else: 
                print(f"[{self.orchestrator_name}] Nodo {w_name} già completato in checkpoint. Salto inferenza.")
        results_lock = threading.Lock()
        connessioni_attive = []
        connessioni_lock = threading.Lock()
        active_worker_names = list(worker_names)
        
        MAX_RETRIES_PER_TASK = 3
        task_retries = {}
        failed_tasks = set()

        def inference_worker_consumer(w_name):
            w_info = available_workers[w_name]
            worker_conn = None
            try:
                print(f" [Federated RPC -> {w_name}] Connessione al nodo privato per inferenza...")
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
                while True:
                    try:
                        task_id, target_worker, forest_bytes = task_queue.get(timeout=1)
                    except queue.Empty:
                        break
                    print(f" [Federated RPC -> {w_name}] Task {task_id}: Inferenza con foresta globale...")
                    if target_worker != w_name:
                        task_queue.put((task_id, target_worker, forest_bytes))
                        task_queue.task_done()
                        time.sleep(0.5)  # Piccola pausa per evitare busy waiting
                        continue
                    print(f" [Federated RPC -> {w_name}] Task {task_id}: Invio foresta globale per inferenza...")
                    try:    
                        raw_response = worker_conn.root.exposed_predict_subset_forest(serialized_trees=forest_bytes,serialized_X_test=None)
                        worker_result = pickle.loads(obtain(raw_response))
                        with results_lock:
                            results_chunks.append((w_name, worker_result))
                            chunks_snapshot = list(results_chunks)
                        try:
                            with open(inference_cp_path, "wb") as f_chk:
                                pickle.dump(chunks_snapshot, f_chk)
                        except Exception as e_fs:
                                print(f" [ERRORE CHECKPOINT] Impossibile aggiornare file di checkpoint inferenza: {e_fs}")
                        print(f" [Federated RPC <- {w_name}] Task {task_id} completato con successo. Risultati salvati in checkpoint.")
                        task_queue.task_done()
                        
                        
                    except Exception as e:
                        print(f" [FALLIMENTO NODO FEDERATO] Il client {w_name} è disconnesso o ha fallito: {e}")
                    
                        retries = task_retries.get(task_id, 0)+1
                        task_retries[task_id] = retries
                        if retries > MAX_RETRIES_PER_TASK:
                            print(f" [FALLIMENTO NODO FEDERATO] Task {task_id} ha superato il numero massimo di tentativi. Segnalo come fallito.")
                            failed_tasks.add(task_id)
                            task_queue.task_done()
                        else:
                            print(f" [Federated RPC <- {w_name}] Task {task_id} reinserito in coda per un altro nodo federato. Tentativo {retries}/{MAX_RETRIES_PER_TASK}.")
                            task_queue.put((task_id, target_worker, forest_bytes))
                            
                        with results_lock:
                            if w_name in active_worker_names:
                                active_worker_names.remove(w_name)
                        
                        break
            except Exception as conn_err:
                print(f" [ERRORE GENERALE] Durante l'esecuzione del task {task_id} sul nodo {w_name}: {conn_err}")
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
                    except: pass
        rpc_start_time = time.perf_counter()
        threads = []
        for name in worker_names:
            t = threading.Thread(target=inference_worker_consumer, args=(name,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if failed_tasks:
            raise RuntimeError(f"Inferenza federata parziale: {len(failed_tasks)} nodi non hanno risposto.")
        if not task_queue.empty() and len(active_worker_names) == 0:
            raise RuntimeError("Task in coda orfani: tutti i nodi federati sono crashati.")

        # Chiusura connessioni residue
        with connessioni_lock:
            for conn in connessioni_attive:
                try:
                    conn.close()
                except Exception:
                    pass

        rpc_inference_time = time.perf_counter() - rpc_start_time
        total_inference_time = time.perf_counter() - inference_start_time

        # 7. AGGREGAZIONE PESATA DEI RISULTATI LOCALI
        # Nel federato non esiste una matrice centralizzata di predizioni:
        # ogni nodo ha valutato la foresta sui propri campioni privati.
        # Aggreghiamo pesando per numero di campioni locali (federated pooling).
        self._print_and_validate_metrics(
            results_chunks=results_chunks,
            tree_type=tree_type,
            job_id=job_id,
            total_inference_time=total_inference_time,
            rpc_inference_time=rpc_inference_time
        )
                    
    def _print_and_validate_metrics(
        self,
        results_chunks: list,
        tree_type: str,
        job_id: str,
        total_inference_time: float,
        rpc_inference_time: float
    ):
        """
        Calcola e stampa le metriche aggregate federata (weighted pooling per n_campioni).
        A differenza del centralizzato (che ha una matrice unica di predizioni),
        qui aggreghiamo i risultati locali di ogni nodo pesandoli per dimensione del shard.
        """
        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE PRESTAZIONI MODELLO FEDERATO FAULT-TOLERANT (JOB: {job_id[:8]})")
        print("═" * 75)
        print(f"  TEMPO TOTALE DI INFERENZA:              {total_inference_time:.4f} secondi")
        print(f"  TEMPO INFERENZA DISTRIBUITA RPC:        {rpc_inference_time:.4f} secondi")
        print("═" * 75 + "\n")

        total_samples = 0
        # Accumulatori pesati per classificazione
        acc_tot = prec_tot = rec_tot = f1_tot = 0.0
        # Accumulatori pesati per regressione
        mae_tot = mse_tot = r2_tot = 0.0

        for w_name, result in results_chunks:
            y_pred = np.array(result["y_pred"])
            y_true = np.array(result["y_true"])
            n = result.get("n_samples", len(y_true))
            total_samples += n

            if tree_type == "classifier":
                acc_tot  += accuracy_score(y_true, y_pred) * n
                prec_tot += precision_score(y_true, y_pred, average="weighted", zero_division=0) * n
                rec_tot  += recall_score(y_true, y_pred, average="weighted", zero_division=0) * n
                f1_tot   += f1_score(y_true, y_pred, average="weighted", zero_division=0) * n
            else:
                mae_tot += mean_absolute_error(y_true, y_pred) * n
                mse_tot += mean_squared_error(y_true, y_pred) * n
                r2_tot  += r2_score(y_true, y_pred) * n

        if total_samples == 0:
            print(f"[{self.orchestrator_name}] Nessun campione raccolto dai nodi. Metriche non calcolabili.")
            return

        if tree_type == "classifier":
            print(f"  Tipo di Modello:                        CLASSIFICATORE (FEDERATO)")
            print(f"  Testing Set Globale (nodi sommati):     {total_samples} campioni")
            print("-" * 75)
            print(f"  ACCURACY MEDIA PESATA SUI NODI:         {(acc_tot  / total_samples) * 100:.2f} %")
            print(f"  PRECISION MEDIA PESATA SUI NODI:        {(prec_tot / total_samples) * 100:.2f} %")
            print(f"  RECALL MEDIA PESATA SUI NODI:           {(rec_tot  / total_samples) * 100:.2f} %")
            print(f"  F1-SCORE MEDIO PESATO SUI NODI:         {(f1_tot   / total_samples) * 100:.2f} %")
        else:
            print(f"  Tipo di Modello:                        REGRESSORE (FEDERATO)")
            print(f"  Testing Set Globale (nodi sommati):     {total_samples} campioni")
            print("-" * 75)
            print(f"  MAE MEDIO PESATO SUI NODI:              {mae_tot / total_samples:.4f}")
            print(f"  MSE MEDIO PESATO SUI NODI:              {mse_tot / total_samples:.4f}")
            print(f"  R2-SCORE MEDIO PESATO SUI NODI:         {r2_tot  / total_samples:.4f}")

        print("═" * 75 + "\n")

        # ─────────────────────────────────────────────────────────────────────────
        # CHECKPOINT (override del centralizzato)
        # ─────────────────────────────────────────────────────────────────────────

    def _get_trees_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/federated_trees_{job_id}.pkl"
        return f"./.local_storage/federated_trees_{job_id}.pkl"

    def _get_inference_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/federated_inference_{job_id}.pkl"
        return f"./.local_storage/federated_inference_{job_id}.pkl"

    def _load_inference_checkpoint(self, job_id: str) -> list:
        path = self._get_inference_checkpoint_path(job_id)
        if self.environment == "local" and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    chunks = pickle.load(f)
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Caricati {len(chunks)} risultati locali dal checkpoint.")
                return chunks
            except Exception as e:
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Errore nel caricamento: {e}")
        return []

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int, alberi_reali: list = None):
        """
        Override: estende il checkpoint della classe base aggiungendo il salvataggio
        fisico degli alberi federati aggregati (identico al centralizzato).
        """
        # 1. Checkpoint logico sul DB (via classe base)
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

        # 2. Checkpoint fisico degli alberi su disco/S3
        if alberi_reali is not None and len(alberi_reali) > 0:
            checkpoint_trees_path = self._get_trees_checkpoint_path(job_id)
            try:
                with open(checkpoint_trees_path, "wb") as f:
                    pickle.dump(alberi_reali, f)
                print(f"[{self.orchestrator_name}] [FEDERATED-CHECKPOINT-FISICO] {len(alberi_reali)} alberi salvati in storage.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE STORAGE] Fallito salvataggio fisico degli alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        """
        Override: rimuove checkpoint fisici degli alberi e dell'inferenza.
        """
        super()._clean_checkpoint(job_id)

        for path in [
            self._get_trees_checkpoint_path(job_id),
            self._get_inference_checkpoint_path(job_id)
        ]:
            if self.environment == "local" and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[{self.orchestrator_name}] [CLEAN OK] Rimosso checkpoint federato: {path}")
                except Exception as e:
                    print(f"[{self.orchestrator_name}] [CLEAN WARN] Impossibile cancellare {path}: {e}")


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()