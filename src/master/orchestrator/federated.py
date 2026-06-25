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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.federated_data_splitter import FederatedDataSplitter
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
        self.worker_shards_paths = {}

    def _resolve_dataset_type(self, payload: dict) -> str:
        """Determina il tipo di dataset basandosi sul payload inviato dal Client."""
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
    

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> int:
        """
        Esegue un macro-step di addestramento federato puro. 
        Distribuisce le richieste sui worker sani che addestreranno gli alberi
        sui propri frammenti di dati locali (shard pre-esistenti).
        """
        job_id = payload.get("job_id")
        self.current_job_id = job_id
        hyperparameters = payload.get("hyperparameters", {})
        
        max_depth = hyperparameters.get("max_depth", None)
        min_samples_split = hyperparameters.get("min_samples_split", 2)
        is_classification = payload.get("task_type", "classification") == "classification"
        dataset_type = self._resolve_dataset_type(payload)

        print(f"\n[{self.orchestrator_name}] === AVVIO MACRO-STEP TRAINING FEDERATO ({start_alberi} -> {target_alberi} alberi) ===")

        # 1. Rilevamento dei worker disponibili nella rete federata
        registry = ServiceRegistry()
        active_worker_names = registry.get_active_workers()
        if not active_worker_names:
            print(f"[{self.orchestrator_name}] [ERRORE CRITICO] Nessun nodo worker attivo registrato. Sospendo lo step.")
            return start_alberi

        print(f"[{self.orchestrator_name}] Nodi federati attivi rilevati: {active_worker_names}")

        # 2. Ripristino dello stato precedente (Cache Checkpoint Fisico degli Alberi)
        all_trained_trees = []
        checkpoint_trees_path = self._get_trees_checkpoint_path(job_id)
        if start_alberi > 0:
            if os.path.exists(checkpoint_trees_path):
                try:
                    with open(checkpoint_trees_path, "rb") as f:
                        all_trained_trees = pickle.load(f)
                    print(f"[{self.orchestrator_name}] [FEDERATED-RESUME] Ripristinati {len(all_trained_trees)} alberi dal checkpoint fisico.")
                except Exception as e:
                    print(f"[{self.orchestrator_name}] [FEDERATED-RESUME WARN] Errore ripristino checkpoint: {e}")
                    all_trained_trees = []

        if len(all_trained_trees) != start_alberi:
            print(f"[{self.orchestrator_name}] [FEDERATED-WARN] Disallineamento cache ({len(all_trained_trees)}) vs start_alberi ({start_alberi}). Allineamento forzato.")
            start_alberi = len(all_trained_trees)

        alberi_da_generare_in_questo_step = target_alberi - start_alberi
        if alberi_da_generare_in_questo_step <= 0:
            print(f"[{self.orchestrator_name}] Target già soddisfatto per questo blocco.")
            return start_alberi

        # 3. Configurazione della Coda dei Sotto-Task (Bilanciamento del carico)
        CHUNK_SIZE = max(1, alberi_da_generare_in_questo_step // len(active_worker_names))
        task_queue = queue.Queue()
        task_id_counter = 1

        sub_start = start_alberi
        worker_cycle_list = list(active_worker_names)
        worker_idx = 0

        while sub_start < target_alberi:
            sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
            assigned_worker = worker_cycle_list[worker_idx % len(worker_cycle_list)]
            
            # Offset assoluto del seed per garantire consistenza matematica deterministica
            task_seed = seed + sub_start
            
            # Inseriamo il task specificando a quale worker assegnarlo idealmente in base alla topologia
            task_queue.put((task_id_counter, assigned_worker, sub_start, sub_end, task_seed))
            
            task_id_counter += 1
            worker_idx += 1
            sub_start = sub_end

        print(f"[{self.orchestrator_name}] Coda dei task configurata con {task_queue.qsize()} elementi distributivi.")
        trees_lock = threading.Lock()

        # 4. Motore di Consumo Multi-Thread
        def thread_consumer():
            while True:
                # Condizione di uscita sicura coordinata dal lock globale
                with trees_lock:
                    if len(all_trained_trees) >= target_alberi:
                        break

                try:
                    task = task_queue.get(timeout=2)
                except queue.Empty:
                    break

                task_id, worker_name, s_start, s_end, t_seed = task
                trees_to_fit = s_end - s_start

                # Controllo dinamico dello stato di salute del cluster
                current_active_workers = registry.get_active_workers()
                if worker_name not in current_active_workers:
                    print(f"[{self.orchestrator_name}-Thread] [FAILOVER] Il worker designato ({worker_name}) è offline.")
                    if not current_active_workers:
                        print(f"[{self.orchestrator_name}-Thread] [CRITICO] Nessun worker disponibile nel cluster. Abort task.")
                        task_queue.put(task)
                        task_queue.task_done()
                        break
                    
                    # Spostamento dinamico della richiesta (Failover) su un altro nodo attivo
                    new_worker = current_active_workers[0]
                    print(f"[{self.orchestrator_name}-Thread] [FAILOVER] Reindirizzo il Task {task_id} sul worker superstite: {new_worker}")
                    task_queue.put((task_id, new_worker, s_start, s_end, t_seed))
                    task_queue.task_done()
                    continue

                # Connessione RPC ed Addestramento Locale sul Worker
                try:
                    worker_info = registry.get_worker_connection_info(worker_name)
                    if not worker_info:
                        raise ConnectionError(f"Metadati di connessione non trovati per il worker {worker_name}")

                    print(f"[{self.orchestrator_name}-Thread] Invio Task {task_id} ({trees_to_fit} alberi: {s_start}-{s_end}) al nodo federato: {worker_name}")
                    
                    conn = rpyc.classic.connect(worker_info["host"], worker_info["port"], timeout=60)
                    worker_service = conn.root

                    # Invocazione federata: Non passiamo alcun dataset path assoluto del Master!
                    # Il worker cercherà internamente il suo file pre-assegnato (es. train_shard.csv nella sua cache/cartella)
                    remote_trees_serialized = worker_service.train_federated_shard(
                        shard_path=None,  # il worker sa dove sono i suoi dati
                        n_estimators=trees_to_fit,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        is_classification=is_classification,
                        random_state=t_seed,
                        dataset_type=dataset_type
                    )

                    remote_trees = pickle.loads(remote_trees_serialized)
                    extracted_trees = obtain(remote_trees)

                    with trees_lock:
                        all_trained_trees.extend(extracted_trees)
                        current_count = len(all_trained_trees)
                    
                    print(f"[{self.orchestrator_name}-Thread] [RPC <- {worker_name}] Task {task_id} completato. Ricevuti {len(extracted_trees)} alberi.")
                    
                    # Scrittura atomica del checkpoint logico e salvataggio file pickle fisico
                    self._save_checkpoint(job_id, current_count, payload.get("retries", 0), seed)
                    checkpoint_trees_path = self._get_trees_checkpoint_path(job_id)
                    with trees_lock:
                        with open(checkpoint_trees_path, "wb") as f:
                            pickle.dump(all_trained_trees, f)
                    
                    conn.close()
                    task_queue.task_done()

                except Exception as e:
                    print(f"[{self.orchestrator_name}-Thread] [ERRORE RPC] Fallimento del worker federato {worker_name} sul Task {task_id}: {e}")
                    # Riposizionamento istantaneo in coda in totale allineamento con la logica centralizzata
                    task_queue.put(task)
                    task_queue.task_done()
                    time.sleep(2)

        # 5. Esecuzione Multi-Thread in parallelo
        num_threads = max(1, len(active_worker_names))
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=thread_consumer)
            t.daemon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return len(all_trained_trees)
    
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

    def _get_trees_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/federated_trees_{job_id}.pkl"
        return f"./.local_storage/checkpoints/federated_trees_{job_id}.pkl"

    def _get_inference_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://my-cluster-datasets-bucket/checkpoints/inference_chunks_{job_id}.pkl"
        return f"./.local_storage/inference_chunks_{job_id}.pkl"

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
        
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

        if alberi_reali is not None and len(alberi_reali) > 0:
            checkpoint_trees_path = self._get_trees_checkpoint_path(job_id)
            try:
                os.makedirs(os.path.dirname(checkpoint_trees_path), exist_ok=True)
                with open(checkpoint_trees_path, "wb") as f:
                    pickle.dump(alberi_reali, f)
                print(f"[{self.orchestrator_name}] [FEDERATED-CHECKPOINT-FISICO] {len(alberi_reali)} alberi archiviati.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE CHECKPOINT] Impossibile persistere gli alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        
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
                    print(f"[{self.orchestrator_name}] [CLEAN WARN] Errore cancellazione {path}: {e}")


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()