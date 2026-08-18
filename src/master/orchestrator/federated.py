import fcntl
import json
import pickle
import os
import random
import socket
import threading
import time
import traceback
from botocore.exceptions import ClientError
import boto3
import rpyc
import numpy as np
import re

from rpyc.utils.classic import obtain
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.dataset.checkpoint_dao import CheckpointDAOFactory
from src.shared.utilities.federated_data_splitter import FederatedDataSplitter
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig

BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")

class FederatedOrchestrator(BaseOrchestrator):
    
    def __init__(self, orchestrator_name: str = None, num_workers: int = None):
        self.cfg = SystemConfig()
        self.num_workers = num_workers or int(os.environ.get("NUM_WORKERS", getattr(self.cfg, "num_workers", 3)))
        name = orchestrator_name or f"Orchestrator-Federato-{socket.gethostname()}"
        
        super().__init__(
            orchestrator_name=name,
            queue_name=self.cfg.sqs_federated_queue
        )
        self.chunk_sent_event = threading.Event()
        self.current_job_id = None
        self.checkpoint_dao = CheckpointDAOFactory.get_dao(self.environment)
        self.worker_wait_timeout = float(os.environ.get("FED_WORKER_WAIT_TIMEOUT_SECONDS", 0))

        # Cache in-memoria (solo per QUESTA istanza di processo) degli alberi
        # già addestrati per un dato job. Serve esclusivamente a evitare una
        # GET S3 ridondante quando il round successivo viene gestito dalla
        # STESSA istanza orchestratore. NON sostituisce mai il checkpoint
        # fisico su S3, che resta l'unica fonte di verità condivisa: se
        # un'altra istanza (nuovo leader dopo un fault) subentra, questa
        # cache sarà vuota/non coerente e si procederà comunque con un
        # reload reale da S3 (vero FAILOVER-RESUME), garantendo il failover.
        self._trees_cache = {}


    def _ensure_local_bootstrap(self, payload: dict):
        """
        Esegue il bootstrap dei file CSV LOCALI se siamo in ambiente 'local'.
        Usa fcntl.flock per garantire mutua esclusione assoluta a livello di File System
        ed evitare race condition tra repliche concorrenti.
        """
        if self.environment != "local":
            print(f"[{self.orchestrator_name}] Ambiente Cloud/AWS rilevato. Bootstrap locale saltato.")
            return
        datasetype = self._resolve_dataset_type(payload)
        lock_dir = "./.local_storage"
        os.makedirs(lock_dir, exist_ok=True)
        bootstrap_mutex = os.path.join(lock_dir, "bootstrap_data.mutex")

        # Acquisiamo un lock esclusivo sul file di bootstrap prima di fare qualsiasi controllo o scrittura
        with open(bootstrap_mutex, "a") as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            try:
                num_workers = self.num_workers
                print(f"[{self.orchestrator_name}] [BOOTSTRAP] Controllo shard per {num_workers} worker...")
                
                if datasetype == "synthetic":
                    print(f"[{self.orchestrator_name}] Dataset SINTETICO rilevato. Il bootstrap e lo sharding sono delegati autonomamente ai singoli worker.")
                    return
                data_folder = getattr(self.cfg, "dataset_path", None)
                if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
                    data_folder = "./dataset_cache" if os.path.exists("./dataset_cache") else "./data"
                
                shards_esistenti = True
                base_cache_dir = "./workers_cache"
                for i in range(1, num_workers + 1):
                    dir_padded = os.path.join(base_cache_dir, f"Worker-Locale-{i:02d}")
                    dir_unpadded = os.path.join(base_cache_dir, f"Worker-Locale-{i}")
                    
                    train_p = os.path.join(dir_padded, "train_shard.csv")
                    test_p = os.path.join(dir_padded, "test_shard.csv")
                    train_up = os.path.join(dir_unpadded, "train_shard.csv")
                    test_up = os.path.join(dir_unpadded, "test_shard.csv")
                    
                    if not ((os.path.exists(train_p) and os.path.exists(test_p)) or 
                            (os.path.exists(train_up) and os.path.exists(test_up))):
                        shards_esistenti = False
                        break
                
                if shards_esistenti:
                    print(f"[{self.orchestrator_name}] [BOOTSTRAP] Shard già presenti su disco. Salto il ricalcolo.")
                else:
                    print(f"[{self.orchestrator_name}] [BOOTSTRAP] Shard incompleti o assenti. Avvio Generazione...")
                    data_loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=0.05, dataset_seed=123)
                    splitter = FederatedDataSplitter(target_column="Label", test_size=0.20, random_state=123)
                    splitter.split_and_shard(data_loader, num_workers=num_workers, environment="local")
                    print(f"[{self.orchestrator_name}] [BOOTSTRAP OK] Shard reali distribuiti nelle cartelle locali dei Worker.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [BOOTSTRAP WARN] Fallimento durante il bootstrap locale: {e}")
            finally:
                fcntl.flock(mutex, fcntl.LOCK_UN)

    def _ensure_aws_bootstrap(self, payload: dict):
        """
        Verifica (senza generarli) che gli shard siano già stati provisionati
        su S3 per l'ambiente AWS. La generazione/upload NON avviene più qui:
        è responsabilità di uno script di provisioning standalone
        (scripts/provision_federated_shards.py), eseguito UNA VOLTA, PRIMA di
        avviare master e worker — coerente con l'idea che, in un vero
        scenario federato, i dati risiedono già sui nodi quando il sistema
        parte, non vengono generati/distribuiti reattivamente durante un job.
        """
        datasetype = self._resolve_dataset_type(payload)
        if datasetype == "synthetic":
            print(f"[{self.orchestrator_name}] Dataset SINTETICO rilevato. Nessun controllo shard necessario "
                  f"(generato autonomamente da ogni worker).")
            return

        num_workers = self.num_workers
        s3_client = boto3.client("s3")

        print(f"[{self.orchestrator_name}] [CHECK AWS] Verifica provisioning shard su S3 per {num_workers} worker...")
        mancanti = []
        for i in range(1, num_workers + 1):
            for fname in ("train_shard.csv", "test_shard.csv"):
                key = f"federated_shards/worker_{i}/{fname}"
                try:
                    s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                except ClientError:
                    mancanti.append(key)

        if mancanti:
            raise RuntimeError(
                f"[{self.orchestrator_name}] Provisioning AWS incompleto: mancano {len(mancanti)} shard su S3 "
                f"(bucket '{BUCKET_NAME}'), es. {mancanti[:3]}. Esegui "
                f"'python -m scripts.provision_federated_shards' prima di avviare il cluster."
            )
        print(f"[{self.orchestrator_name}] [CHECK AWS OK] Tutti gli shard richiesti sono presenti su S3.")
        
    def _perform_active_recovery(self):
        """Innesca il bootstrap locale subito dopo la conquista del lock di leadership."""
        
        super()._perform_active_recovery()

    def _infer_worker_index(self, w_name: str, fallback_idx: int) -> int:
        marker = re.search(r"WIDX(\d+)", w_name)
        if marker:
            return int(marker.group(1))
        print(f"[{self.orchestrator_name}] [WARN] Impossibile derivare un indice stabile dal nome "
            f"'{w_name}'. Fallback sulla posizione nella lista ({fallback_idx}).")
        return fallback_idx

    def _resolve_dataset_type(self, payload: dict) -> str:
        """Determina il tipo di dataset basandosi sul payload inviato dal Client."""
        dataset_type = payload.get("dataset_type")
        if dataset_type:
            return str(dataset_type).strip().lower()
        return "real"
    
    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> int:
        """
        Invia la richiesta a ciascun worker attivo per il proprio shard locale.
        Se un worker fallisce, viene registrato il dropout e si prosegue con i rimanenti.
        Alla fine, se gli alberi totali superano il target (over-provisioning), applica lo scarto uniforme.
        """
        self.current_job_id = payload.get("job_id")
        
        checkpoint_trees_path = self._resolve_trees_checkpoint_path(self.current_job_id)
        if self.environment == "aws":
            self._ensure_aws_bootstrap(payload)
        else:
            self._ensure_local_bootstrap(payload)
            os.makedirs("./.local_storage", exist_ok=True)

        
        if start_alberi == 0 and self.checkpoint_dao.exists(checkpoint_trees_path):
            self.checkpoint_dao.delete(checkpoint_trees_path)
        if start_alberi == 0:
            self._trees_cache.pop(self.current_job_id, None)
        all_trained_trees = []
        if start_alberi > 0:
            cached = self._trees_cache.get(self.current_job_id)
            if cached is not None and len(cached) == start_alberi:
                # Stessa istanza, stesso job: nessun fault, è solo il round successivo
                # nello stesso processo. Riusiamo la lista già in memoria, niente GET S3.
                print(f"\n[{self.orchestrator_name}] [STATE-SYNC] Continuazione round nella stessa istanza "
                      f"({start_alberi} alberi già in memoria). Nessun reload da storage necessario.")
                all_trained_trees = cached
            else:
                # Cache assente o non coerente con start_alberi: questa istanza non ha
                # memoria diretta del progresso richiesto (riavvio dopo crash, o nuovo
                # leader subentrato dopo un fault di un'altra istanza). Il checkpoint
                # fisico su S3 (fonte di verità condivisa) è l'unico modo sicuro per
                # recuperare lo stato: qui avviene il vero, garantito, recovery cross-istanza.
                print(f"\n[{self.orchestrator_name}] [FAILOVER-RESUME] Nessuna cache locale valida per "
                      f"start_alberi = {start_alberi}. Ripristino checkpoint fisico da storage condiviso...")
                if self.checkpoint_dao.exists(checkpoint_trees_path):
                    try:
                        # NOTA: il checkpoint viene sempre salvato come lista COMPLETA e aggiornata
                        # (overwrite, non append) tramite checkpoint_dao.save(): un singolo load()
                        # restituisce già tutti gli alberi, senza bisogno di ricostruire nulla a mano.
                        all_trained_trees = self.checkpoint_dao.load(checkpoint_trees_path)
                        print(f"[{self.orchestrator_name}] [OK] Ripristinati con successo {len(all_trained_trees)} alberi reali dal checkpoint.")
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
            return len(all_trained_trees)
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
            tree_type = hp.get("tree_type", "classifier")

            CHUNK_SIZE = int(np.ceil(total_step_trees /num_workers ))
            print(f"[{self.orchestrator_name}] Calcolo dinamico: {num_workers} worker rilevati -> CHUNK_SIZE impostata a {CHUNK_SIZE} alberi per task.")

            assigned_tasks = {}
            sub_start = start_alberi
            task_id_counter = start_alberi + 1
            # riceve UN SOLO chunk, di sua esclusiva proprietà. Se il worker muore mentre
            # lo sta processando, il chunk NON viene ripreso da nessun altro worker:
            # il thread dedicato smette di ritentare la RPC e resta in attesa che quello
            # stesso worker ricompaia nel ServiceRegistry, poi riprova lo stesso task.
            for w_name in worker_names:
                if sub_start >= target_alberi:
                    break
                sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
                task_seed = seed + sub_start
                assigned_tasks[w_name] = (task_id_counter, sub_start, sub_end, task_seed)
                task_id_counter += 1
                sub_start = sub_end
                
            feature_selezionate = (None if self.environment == "aws" else self.select_from_config(self._resolve_dataset_type(payload)))
            results_lock = threading.Lock()
            checkpoint_time_accum = [0.0]
            RETRY_WAIT_SECONDS = 10
            # Reset dell'evento (già usato in fase di inferenza): qui serve a far sì
            # che i test di fault injection possano attendere in modo affidabile il
            # momento in cui il PRIMO task di training viene davvero inviato a un
            # worker, invece di limitarsi a un'attesa temporale fissa.
            self.chunk_sent_event.clear()
            def contact_worker(w_name, idx):
                task = assigned_tasks.get(w_name)
                if task is None:
                    return 
                task_id, start_t, end_t, chunk_seed = task
                quota_chunk = end_t - start_t
                wait_started_at = time.perf_counter()
                while True:
                    while True:
                        available_now = ServiceRegistry.get_available_workers(self.environment)
                        if w_name in available_now:
                            w_info = available_now[w_name]
                            break
                        
                        if self.worker_wait_timeout > 0 and (time.perf_counter() - wait_started_at) > self.worker_wait_timeout:
                            print(f"[{self.orchestrator_name}] [TIMEOUT] Worker '{w_name}' non tornato disponibile "
                                f"entro {self.worker_wait_timeout:.0f}s. Task {task_id} ({quota_chunk} alberi) "
                                f"ABBANDONATO per questo round. Verrà ritentato al prossimo step con i worker rimasti.")
                            return  # rinuncia al chunk per questo step, senza bloccare gli altri thread

                        print(f"[{self.orchestrator_name}] [WAIT] Worker '{w_name}' non raggiungibile. "
                              f"Il suo Task {task_id} ({quota_chunk} alberi) resta in attesa: "
                              f"nessun altro worker lo prenderà in carico.")
                        time.sleep(RETRY_WAIT_SECONDS)
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
                        with self.connessioni_lock:
                            self.connessioni_attive.append(worker_conn)
                        print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="PROCESSING")
                        self.chunk_sent_event.set()
                        result_raw = worker_conn.root.exposed_train_local_federated_forest(
                            job_id=self.current_job_id,
                            dataset_type=self._resolve_dataset_type(payload),
                            n_estimators_local=quota_chunk,
                            worker_index=idx,
                            hyperparameters={
                                **hp, 
                                "random_state": chunk_seed + (idx * 1000),
                                "feature_selezionate": feature_selezionate,
                                
                            },
                        )
                        result_trees = pickle.loads(obtain(result_raw))
                        with results_lock:
                            all_trained_trees.extend(result_trees)
                            current_total = len(all_trained_trees)
                            try:
                                t_chk_start = time.perf_counter()
                                self.checkpoint_dao.save(checkpoint_trees_path, all_trained_trees)
                                checkpoint_time_accum[0] += time.perf_counter() - t_chk_start
                                # Manteniamo la cache di istanza allineata SOLO dopo che il
                                # salvataggio fisico su storage condiviso è andato a buon fine,
                                # così non è mai "più avanti" della fonte di verità persistita.
                                self._trees_cache[self.current_job_id] = list(all_trained_trees)
                                print(f"   [RPC <- {w_name}] [CHECKPOINT FS OK] Task {task_id} archiviato. Progressivo in RAM/Storage: {current_total} alberi.")
                            except Exception as e_fs:
                                print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere gli alberi parziali su file: {e_fs}")
                            if hasattr(self, 'state_manager') and self.state_manager:
                                try:
                                    self.state_manager.update_request_status(
                                        job_id=self.current_job_id,
                                        status="PROCESSING",
                                        orchestrator_id=self.orchestrator_name,
                                        retries=payload.get("retries", 0),
                                        base_random_state=seed,
                                        alberi_addestrati=current_total,
                                    )
                                except Exception as e_db:
                                    print(f"   [ERRORE] Impossibile inviare l'heartbeat di stato a DynamoDB: {e_db}")
                            
                        print(f"   [RPC <- {w_name}] Task {task_id} completato. Ricevuti {len(result_trees)} alberi.")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="COMPLETED")
                        return  # task di questo worker concluso, il thread termina
                    except Exception as e:
                        print(f"   [ERRORE RPC] Fallimento o disconnessione del worker {w_name} durante il Task {task_id}: {e}")
                        print(f"[{self.orchestrator_name}-Thread] Task {task_id} NON viene riassegnato ad altri worker. "
                              f"In attesa che '{w_name}' si riavvii per riprendere lo stesso chunk.")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="WAITINGFORWORKER")
                        time.sleep(RETRY_WAIT_SECONDS)
                        continue  # nessun task_queue.put(): il chunk resta di proprietà esclusiva di w_name
                    finally:
                        if worker_conn:
                            with self.connessioni_lock:
                                if worker_conn in self.connessioni_attive:
                                    self.connessioni_attive.remove(worker_conn)
                            try:
                                worker_conn.close()
                            except Exception:
                                pass
            threads = []
            for i, worker_name in enumerate(worker_names, start=1):
                stable_idx = self._infer_worker_index(worker_name, i)
                t = threading.Thread(target=contact_worker, args=(worker_name, stable_idx))
                threads.append(t)
                t.start()
 
            for t in threads:
                t.join()
            print(f"[DEBUG] Tempo totale speso in I/O di checkpoint: {checkpoint_time_accum[0]:.2f}s")   
 
            if not all_trained_trees:
                raise RuntimeError("Tutti i nodi interessati sono falliti. Nessun albero raccolto per questo Job.")
 
            if len(all_trained_trees) > target_alberi:
                print(f"[{self.orchestrator_name}] [SCARTO UNIFORME] Trovati {len(all_trained_trees)} alberi. Riduzione casuale a quota {target_alberi}.")
                collected_trees = random.sample(all_trained_trees, target_alberi)
            else:
                print(f"[{self.orchestrator_name}] Raccolti in totale {len(all_trained_trees)} alberi dai worker superstiti.")
                collected_trees = all_trained_trees
            
            final_count = self._reconstruct_and_save_global_model(collected_trees, tree_type)
            self._save_checkpoint(self.current_job_id, final_count, payload.get("retries", 0), seed, alberi_reali=collected_trees)
            return final_count

    def _execute_inference_step(self, payload: dict) -> dict:
        print(f"\n[{self.orchestrator_name}] == AVVIO VALIDAZIONE FEDERATA DISTRIBUITA ==")
        job_id = payload.get("job_id")
        hyperparameters = payload.get("hyperparameters", {})
        tree_type = hyperparameters.get("tree_type", "classifier")

        inference_start_time = time.perf_counter()

        model_path = self._resolve_model_path(job_id)
        if not self.checkpoint_dao.exists(model_path):
            raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")
        print(f"[{self.orchestrator_name}] Caricamento della foresta globale da {model_path}...")
        global_model = self.checkpoint_dao.load(model_path)

        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta caricata. Numero totale di alberi: {total_trees}")

        available_workers = ServiceRegistry.get_available_workers(self.environment)
        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        if num_workers == 0:
            raise RuntimeError("Nessun worker disponibile per l'inferenza federata.")
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        forest_bytes = pickle.dumps(all_trees)
        feature_selezionate = (
            None if self.environment == "aws"
            else self.select_from_config(self._resolve_dataset_type(payload))
        )

        y_pred_global = []
        y_true_global = []
        y_probs_global = []
        total_samples_ref = [0]
        failed_workers = set()
        self.chunk_sent_event.clear()   
        results_lock = threading.Lock()
        INF_RETRY_WAIT_SECONDS = 10

        def validate_worker(w_name, idx):
            wait_started_at = time.perf_counter()
            while True:
                while True:
                    available_now = ServiceRegistry.get_available_workers(self.environment)
                    if w_name in available_now:
                        w_info = available_now[w_name]
                        break
 
                    if self.worker_wait_timeout > 0 and (time.perf_counter() - wait_started_at) > self.worker_wait_timeout:
                        print(f"[{self.orchestrator_name}] [TIMEOUT INF] Worker '{w_name}' non è tornato disponibile "
                              f"entro {self.worker_wait_timeout:.0f}s. I suoi campioni vengono ESCLUSI dalla metrica "
                              f"finale (status PARTIAL), non richiesti ad altri worker.")
                        with results_lock:
                            failed_workers.add(w_name)
                        return
                    print(f"[{self.orchestrator_name}] [WAIT INF] Worker '{w_name}' non raggiungibile. "
                          f"La sua validazione resta in attesa: nessun altro worker userà il suo test-shard.")
                    time.sleep(INF_RETRY_WAIT_SECONDS)
                conn = None
                try:
                    print(f" [RPC INF -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                    conn = rpyc.connect(
                        w_info["host"], w_info["port"],
                        config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": 300}
                    )
                    with self.connessioni_lock:
                        self.connessioni_attive.append(conn)
                    self.chunk_sent_event.set()
 
                    worker_hyperparameters = {
                        **hyperparameters,
                        "dataset_type": self._resolve_dataset_type(payload),
                        "feature_selezionate": feature_selezionate,
                        "tree_type": tree_type,
                    }
                    if tree_type == "classifier" and hasattr(global_model, "classes_"):
                        worker_hyperparameters["global_classes"] = global_model.classes_.tolist()
                    # ----------------------------------------------------------------------------
                    print(f"[{self.orchestrator_name}-InfThread] Invio foresta completa ({total_trees} alberi) a {w_name}...")
                    raw_response = conn.root.exposed_predict_subset_forest(payload=pickle.dumps({
                        "forest": forest_bytes,
                        "job_id": job_id,
                        "worker_index": idx,
                        "hyperparameters": worker_hyperparameters
                    }))
                    worker_data = pickle.loads(obtain(raw_response))
 
                    with results_lock:
                        y_pred_global.extend(worker_data["y_pred"])
                        y_true_global.extend(worker_data["y_true"])
                        total_samples_ref[0] += worker_data["n_samples"]
                        if tree_type == "classifier":
                            # Chiave opzionale: worker meno recenti potrebbero non restituirla ancora.
                            worker_probs = worker_data.get("y_probs")
                            if worker_probs is not None:
                                y_probs_global.extend(worker_probs)
                            else:
                                print(f"[{self.orchestrator_name}] [WARN] Worker '{w_name}' non ha restituito "
                                      f"'y_probs': l'AUC finale sarà None (worker non aggiornato).")
                        print(f"[{self.orchestrator_name}] Validazione completata su '{w_name}' ({worker_data['n_samples']} record).")
                    return  # successo, il thread termina
 
                except Exception as ex:
                    print(f"   [ERRORE INF] Fallimento su '{w_name}': {ex}. In attesa che torni disponibile "
                          f"(la sua validazione NON verrà eseguita da altri worker).")
                    time.sleep(INF_RETRY_WAIT_SECONDS)
                    continue  # torna al ciclo di attesa, stesso worker
                finally:
                    if conn:
                        with self.connessioni_lock:
                            if conn in self.connessioni_attive:
                                self.connessioni_attive.remove(conn)
                        try:
                            conn.close()
                        except Exception:
                            pass
 
        rpc_start_time = time.perf_counter()
        threads = []
        for idx, name in enumerate(worker_names, start=1):
            stable_idx = self._infer_worker_index(name,idx)
            t = threading.Thread(target=validate_worker, args=(name, stable_idx))
            t.start()
            threads.append(t)
 
        for t in threads:
            t.join()
 
        with self.connessioni_lock:
            for conn in self.connessioni_attive:
                try: conn.close()
                except Exception: pass
 
        rpc_inference_time = time.perf_counter() - rpc_start_time
 
        if not y_pred_global:
            print(f"[{self.orchestrator_name}] [ERRORE] Nessun worker ha risposto alla validazione federata.")
            return {}
 
        if failed_workers:
            print(f"[{self.orchestrator_name}] [WARN] {len(failed_workers)} worker non hanno risposto: {failed_workers}. Metriche calcolate sui rimanenti.")
 
        total_inference_time = time.perf_counter() - inference_start_time
 
        y_true_dtype = np.float64 if tree_type == "regressor" else np.int64

        # y_probs è allineato sample-per-sample con y_pred_global/y_true_global SOLO se
        # ogni worker rispondente lo ha fornito (stesso ordine di extend()). In caso contrario
        # l'array sarebbe disallineato: meglio non calcolare l'AUC piuttosto che calcolarlo male.
        y_probs_array = None
        if tree_type == "classifier" and len(y_probs_global) == len(y_pred_global):
            y_probs_array = np.array(y_probs_global, dtype=np.float64)

        # I worker restituiscono già la predizione finale del modello globale sul proprio
        # shard locale (non i voti dei singoli alberi), quindi qui NON si passa da
        # _aggregate_forest_predictions: si calcolano le metriche direttamente.
        metrics = self.calculate_metrics(
            final_predictions=np.array(y_pred_global, dtype=np.float64),
            y_test=np.array(y_true_global, dtype=y_true_dtype),
            tree_type=tree_type,
            y_probs=y_probs_array
        )
        self._save_metrics(job_id, "inference", {
            "job_id": job_id, "mode": "federated", "phase": "inference",
            "tree_type": tree_type, "testing_set_size": total_samples_ref[0],
            "timings": {"total_inference_time": total_inference_time, "rpc_inference_time": rpc_inference_time},
            "metrics": metrics
        })
        if hasattr(self, 'state_manager') and self.state_manager:
            try:
                self.state_manager.update_request_status(
                    job_id=job_id,
                    status="COMPLETED",
                    orchestrator_id=self.orchestrator_name,
                    alberi_addestrati=total_trees,
                )
            except Exception as e_db:
                print(f"   [ERRORE] Impossibile scrivere lo stato COMPLETED su DynamoDB/local: {e_db}")
 
        return {
            "status": "SUCCESS" if not failed_workers else "PARTIAL",
            "testing_set_size": total_samples_ref[0],
            "failed_workers": list(failed_workers),
            "total_inference_time": total_inference_time,
            "rpc_inference_time": rpc_inference_time,
            "metrics": metrics
        } 

    
    def _reconstruct_and_save_global_model(self, all_trained_trees: list, tree_type: str) -> int:
        if not all_trained_trees:
            print(f"[{self.orchestrator_name}] Nessun albero collezionato.")
            return 0

        print(f"[{self.orchestrator_name}] Ricomposizione foresta globale conforme a Scikit-Learn...")
        try:
            n_features = all_trained_trees[0].n_features_in_
            
            if tree_type == "classifier":
                global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                # Stesso fix applicato in centralized.py: classi derivate dagli alberi reali
                # invece di un'assunzione binaria fissa {0, 1}.
                trees_with_classes = [t for t in all_trained_trees if hasattr(t, "classes_")]
                if trees_with_classes:
                    detected_classes = np.unique(np.concatenate([np.asarray(t.classes_) for t in trees_with_classes]))
                else:
                    print(f"[{self.orchestrator_name}] [WARN] Nessun albero espone 'classes_'. Fallback su {{0, 1}}.")
                    detected_classes = np.array([0, 1])
                global_model.classes_ = detected_classes.astype(np.int64)
                global_model.n_classes_ = len(detected_classes)
            else:
                global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
            
            global_model.estimators_ = all_trained_trees
            global_model.n_features_in_ = n_features
            global_model.n_outputs_ = 1
            
            model_path = self._resolve_model_path(self.current_job_id)
            self.checkpoint_dao.save(model_path, global_model)
 
            print(f"[{self.orchestrator_name}] Modello Globale salvato con successo in '{model_path}'.")
            
            
            return len(all_trained_trees)
            
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE AGGREGAZIONE] Fallimento durante l'unione dei sotto-modelli: {e}")
            traceback.print_exc()
            return len(all_trained_trees)
    
        
    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int, alberi_reali: list = None):
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

        if alberi_reali is not None and len(alberi_reali) > 0:
            checkpoint_trees_path = self._resolve_trees_checkpoint_path(job_id)
            try:
                self.checkpoint_dao.save(checkpoint_trees_path, alberi_reali)
                print(f"[{self.orchestrator_name}] Checkpoint alberi salvato in {checkpoint_trees_path}.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE CHECKPOINT] Impossibile salvare checkpoint alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        super()._clean_checkpoint(job_id)
        self._trees_cache.pop(job_id, None)
        checkpoint_trees_path = self._resolve_trees_checkpoint_path(job_id)
        try:
            self.checkpoint_dao.delete(checkpoint_trees_path)
            print(f"[{self.orchestrator_name}] Checkpoint alberi rimosso da {checkpoint_trees_path}.")
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE CLEANUP] Impossibile rimuovere checkpoint alberi: {e}")
 
        inference_cp = self._get_inference_checkpoint_path(job_id)
        try:
            self.checkpoint_dao.delete(inference_cp)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE CLEANUP] Impossibile rimuovere checkpoint inferenza: {e}")
    
    def _resolve_trees_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/checkpoints/checkpoint_trees_{job_id}.pkl"
        return f"./.local_storage/checkpoint_trees_{job_id}.pkl"
 
    def _resolve_model_path(self, job_id: str) -> str:
        """Path del modello globale aggregato, in una sotto-cartella dedicata alla
        modalità federata per evitare collisioni col modello centralizzato in caso
        di job_id riutilizzati tra le due modalità."""
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/federated/model_{job_id}.pkl"
        return os.path.join("./saved_models", f"model_{job_id}.pkl")

    def _get_inference_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/checkpoints/inference_chunks_{job_id}.pkl"
        return f"./.local_storage/inference_chunks_{job_id}.pkl"
    
    def _load_inference_checkpoint(self, job_id: str):
        path = self._get_inference_checkpoint_path(job_id)
        if self.checkpoint_dao.exists(path):
            try:
                chunks = self.checkpoint_dao.load(path)
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Caricati {len(chunks)} chunk di inferenza dal checkpoint.")
                return chunks
            except Exception as e:
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Errore nel caricamento del checkpoint: {e}")
        return []
    
    def select_from_config(self, dataset_type: str = "real"):
        config_filename = f"config_{dataset_type}.json"
        config_path = os.path.join(os.getcwd(), "outputs_baseline", config_filename)
        
        if not os.path.exists(config_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
            config_path = os.path.join(project_root, "outputs_baseline", config_filename)

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
            print(f"[{self.orchestrator_name}] [ATTENZIONE] {config_filename} non trovato in nessuno dei percorsi:")
            print(f"  • {config_path}")
        return None

if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()