import pickle
import os
import socket
import time
import rpyc
import queue
import threading
import traceback
from rpyc.utils.classic import obtain
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import src.shared.utilities.datasplitter
from src.shared.config import SystemConfig
from src.shared.factory import DatasetDAOFactory
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.featureselection import CICIDSFeatureSelector
from src.dataset.checkpoint_dao import CheckpointDAOFactory


TEST_SIZE = 0.2
BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")

RPC_SYNC_TIMEOUT_SECONDS = int(os.environ.get("RPC_SYNC_TIMEOUT_SECONDS", 1800))
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS = int(os.environ.get("RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", 900))

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, orchestrator_name: str = None):
        self.cfg = SystemConfig()
        name = orchestrator_name or f"Orchestrator-Centralizzato-{socket.gethostname()}"

        self.current_job_id = None
        self.train_data_path = None
        self.test_data_path = None
        self.chunk_sent_event = threading.Event()
        self._trees_cache = {}

        super().__init__(
            orchestrator_name=name,
            queue_name=self.cfg.sqs_centralized_queue
        )
        self.checkpoint_dao = CheckpointDAOFactory.get_dao(self.environment)

    def _resolve_dataset_type(self, payload: dict) -> str:
        """Determina il tipo di dataset basandosi sul payload inviato dal Client."""
        dataset_type = payload.get("dataset_type")
        if dataset_type:
            return str(dataset_type).strip().lower()
        return "real"

    def _prepare_data(self, payload: dict, base_seed: int):
        t0 = time.perf_counter()
        job_id = payload.get("job_id", "unknown_job")
        dataset_path = payload.get("dataset_path")
        dataset_type = self._resolve_dataset_type(payload)
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Target" if  tree_type == "regressor" else "Label"

        splitter = src.shared.utilities.datasplitter.StratifiedDataSplitter(target_column=target_col, test_size=TEST_SIZE, random_state=base_seed)

        print(f"\n[{self.orchestrator_name}] Avvio ETL. Tipo: {dataset_type}")

        if dataset_type == "synthetic":
            loader = SyntheticDataLoader(task="regression" if tree_type == "regressor" else "classification", target_column=target_col)
            df_full = loader.load()

            if tree_type == "regressor":
                train_df, test_df = train_test_split(df_full, test_size=TEST_SIZE, random_state=base_seed)
            else:
                train_df, test_df = splitter.split(df_full)
        else:
            if not dataset_path:
                raise ValueError("dataset_path mancante.")
            print(f"[DEBUG] dataset_path ricevuto = {repr(dataset_path)}")
            loader = RawCSVDataLoader(data_url=dataset_path, sample_fraction=0.01, dataset_seed=base_seed)
            df_raw = loader.load()

            # Istanziamo il nuovo preprocessor modificato
            preprocessor = CICIDSPreprocessor(target_column=target_col)
            # ─── FASE 1: BINARIZZAZIONE SUL DATO INTERO ───
            df_binarized = preprocessor.binarize_target(df_raw)
            del df_raw
            # ─── FASE 2: SPLIT STRATIFICATO ADESSO SICURO ───
            print(f"[{self.orchestrator_name}] Esecuzione Split Stratificato...")
            train_df, test_df = splitter.split(df_binarized)
            del df_binarized

            # ─── FASE 3 & 4: PREPROCESAMENTO INDIPENDENTE (Metadata + NaN/inf) ───
            print(f"\n[{self.orchestrator_name}] === PREPROCESSING SUL TRAIN SET ===")
            train_df = preprocessor.process(train_df)

            print(f"\n[{self.orchestrator_name}] === PREPROCESSING SUL TEST SET ===")
            test_df = preprocessor.process(test_df)

        # --- FEATURE SELECTION (Solo Real) ---
        if dataset_type == "real":
            fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
            train_df = fs.fit_transform(train_df)
            test_df = fs.transform(test_df)

        # --- SALVATAGGIO COORDINATO DAI DAO ---
        if self.environment == "aws":
            train_data_path = f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{job_id}.csv"
            test_data_path = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{job_id}.csv"
        else:
            train_data_path = f"./.local_storage/shared_train_{job_id}.csv"
            test_data_path = f"./.local_storage/shared_test_{job_id}.csv"

        print(f"\n[{self.orchestrator_name}] Delega salvataggio a DatasetDAOFactory...")
        try:
            dao = DatasetDAOFactory.get_dao(self.environment)
            dao.save_dataset(path=train_data_path, df=train_df)
            dao.save_dataset(path=test_data_path, df=test_df)
            print(f"[DEBUG TIMING] _prepare_data completato in {time.perf_counter() - t0:.2f}s")
            print(f"[{self.orchestrator_name}] [OK] Dataset di Train e Test archiviati correttamente.")
        except Exception as e:
            raise IOError(f"[{self.orchestrator_name}] Errore critico nel salvataggio dei dataset tramite DAO: {e}")
        self.current_job_id = job_id
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> int:
        """
        Esegue lo step di addestramento distribuito centralizzato.
        Restituisce il numero REALE di alberi totali validati e salvati con successo.
        """
        expected_job_id = payload.get("job_id", "unknown_job")
        # 1. Preparazione dei dati (se non ancora pronti e non presenti su disco)
        if self.train_data_path is None or self.current_job_id != expected_job_id:
            if self.environment == "aws":
                expected_train = f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{expected_job_id}.csv"
                expected_test = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{expected_job_id}.csv"
            else:
                expected_train = f"./.local_storage/shared_train_{expected_job_id}.csv"
                expected_test = f"./.local_storage/shared_test_{expected_job_id}.csv"

            dao = DatasetDAOFactory.get_dao(self.environment)
            if dao.exists(expected_train) and dao.exists(expected_test):
                print(f"[{self.orchestrator_name}] [SHORT-CIRCUIT ETL] Dataset già presente nello storage condiviso. Salto la fase ETL.")
                self.train_data_path = expected_train
                self.test_data_path = expected_test
                self.current_job_id = expected_job_id
            else:
                self._prepare_data(payload, seed)
        checkpoint_trees_path = self._resolve_trees_checkpoint_path(self.current_job_id)
        if self.environment != "aws":
            os.makedirs("./.local_storage", exist_ok=True)
        all_trained_trees = []

        # Pulizia preventiva: se ripartiamo da zero per QUESTO job_id ma esiste
        # già un checkpoint fisico residuo (es. retry manuale con lo stesso id,
        # o rerun dopo una pulizia incompleta), lo scartiamo per evitare che
        # venga riletto per errore da un round successivo (parità con FederatedOrchestrator).
        if start_alberi == 0 and self.checkpoint_dao.exists(checkpoint_trees_path):
            self.checkpoint_dao.delete(checkpoint_trees_path)
        self._trees_cache.pop(self.current_job_id, None) if start_alberi == 0 else None

        # ─── SINCRONIZZAZIONE STATO: SE ABBIAMO GIÀ ALBERI DA UN ROUND PRECEDENTE ───
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
                # memoria diretta del progresso richiesto. Può essere un riavvio dopo
                # crash, oppure un nuovo leader subentrato dopo un fault di un'altra
                # istanza. In entrambi i casi il checkpoint fisico su S3 (fonte di
                # verità condivisa) è l'unico modo sicuro per recuperare lo stato:
                # qui avviene il vero, garantito, recovery cross-istanza.
                print(f"\n[{self.orchestrator_name}] [FAILOVER-RESUME] Nessuna cache locale valida per "
                      f"start_alberi = {start_alberi}. Ripristino checkpoint fisico da storage condiviso...")
                if self.checkpoint_dao.exists(checkpoint_trees_path):
                    try:
                        all_trained_trees = self.checkpoint_dao.load(checkpoint_trees_path)

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

        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)
        tree_type = hp.get("tree_type", "classifier")
        # Fallback allineato ai default "corretti" di RandomForest{Classifier,Regressor}
        # se il manifesto non lo specifica esplicitamente.
        max_features = hp.get("max_features", "sqrt" if tree_type == "classifier" else 1 / 3)
        
        # Caso limite: già finito tutto ma eravamo crashati prima di consolidare
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
            source_info = self.train_data_path

            # 3. CALCOLO DINAMICO DELLA DIMENSIONE DEL CHUNK
            CHUNK_SIZE = int(np.ceil(total_step_trees /num_workers ))
            print(f"[{self.orchestrator_name}] Calcolo dinamico: {num_workers} worker rilevati -> CHUNK_SIZE impostata a {CHUNK_SIZE} alberi per task.")

            # 4. Configurazione della Coda di Sotto-Task locale
            task_queue = queue.Queue()
            sub_start = start_alberi
            task_id_counter = start_alberi + 1

            while sub_start < target_alberi:
                sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
                # Ogni sotto-task associa un seed specifico calcolato sull'offset cumulativo
                task_seed = seed + sub_start
                # Usiamo sub_start come offset assoluto rispetto al seed iniziale del JOB
                task_queue.put((task_id_counter, sub_start, sub_end, task_seed))
                task_id_counter += 1
                sub_start = sub_end

            results_lock = threading.Lock()

            active_worker_names = list(worker_names)

            # Reset dell'evento (già usato in fase di inferenza): qui serve a far sì
            # che i test di fault injection possano attendere in modo affidabile il
            # momento in cui il PRIMO task di training viene davvero inviato a un
            # worker, invece di limitarsi a un'attesa temporale fissa.
            self.chunk_sent_event.clear()

            # 5. Definizione della funzione consumatrice per i thread
            def worker_thread_consumer(w_name):
                w_info = available_workers[w_name]
                worker_conn = None
                try:
                    print(f" [RPC -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                    worker_conn = rpyc.connect(
                        w_info["host"],
                        w_info["port"],
                        config={
                            'allow_pickle': True,
                            'sync_request_timeout': RPC_SYNC_TIMEOUT_SECONDS,
                            'keepalive': True
                        }
                    )
                    with self.connessioni_lock:
                        self.connessioni_attive.append(worker_conn)

                    # ─── Il thread resta attivo finché non raccogliamo la quota di alberi globale ───
                    while len(all_trained_trees) < target_alberi:
                        try:
                            # Timeout breve (2 secondi) per controllare periodicamente lo stato e non restare appesi
                            task_id, start_t, end_t, chunk_seed = task_queue.get(timeout=2)
                        except queue.Empty:
                            # Se la coda è momentaneamente vuota ma mancano alberi al target globale,
                            # un altro worker attivo potrebbe crashare a breve e rimettere un task in coda.
                            # Usciamo solo se l'addestramento è finito o se siamo l'ultimo worker attivo rimasto.
                            with results_lock:
                                total_attuali = len(all_trained_trees)
                                num_worker_attivi = len(active_worker_names)

                            if total_attuali >= target_alberi or num_worker_attivi <= 1:
                                break
                            time.sleep(1)
                            continue

                        quota_chunk = end_t - start_t
                        print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="PROCESSING")
                        try:
                            self.chunk_sent_event.set()

                            result_raw = worker_conn.root.train_subset_forest(
                                source_info=source_info,
                                num_trees=quota_chunk,
                                base_seed=chunk_seed,
                                max_depth=max_depth,
                                tree_type=hp.get("tree_type"),
                                max_features=max_features
                            )

                            # Deserializzazione sicura dei byte trasmessi via rete
                            result_trees = pickle.loads(obtain(result_raw))

                            with results_lock:
                                all_trained_trees.extend(result_trees)
                                current_total = len(all_trained_trees)
                                # SALVATAGGIO FISICO ATOMICO PROGRESSIVO
                                try:
                                    self.checkpoint_dao.save(checkpoint_trees_path, all_trained_trees)
                                    # Manteniamo la cache di istanza allineata SOLO dopo che il
                                    # salvataggio fisico su storage condiviso è andato a buon fine:
                                    # così la cache non è mai "più avanti" della fonte di verità
                                    # persistita, che resta ciò che un'altra istanza rileggerebbe in caso di failover.
                                    self._trees_cache[self.current_job_id] = list(all_trained_trees)
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
                                            base_random_state=seed,
                                            alberi_addestrati=current_total
                                        )
                                    except Exception as e_db:
                                        print(f"   [ERRORE] Impossibile inviare l'heartbeat di stato a DynamoDB: {e_db}")

                            print(f"   [RPC <- {w_name}] Task {task_id} completato. Ricevuti {len(result_trees)} alberi.")
                            self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="COMPLETED")
                            task_queue.task_done()

                        except Exception as e:

                            self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="FAILED")
                            print(f"   [ERRORE RPC] Fallimento o disconnessione del worker {w_name} durante il Task {task_id}: {e}")
                            # FAULT TOLERANCE REALE: Reinserimento immediato del chunk per la fault tolerance
                            task_queue.put((task_id, start_t, end_t, chunk_seed))
                            print(f"[{self.orchestrator_name}-Thread] Task {task_id} riaccodato con successo per il failover.")

                            with results_lock:
                                if w_name in active_worker_names:
                                    active_worker_names.remove(w_name)
                            break  # Il canale RPC con questo worker è saltato, chiudiamo il thread relativo

                except Exception as conn_err:
                    print(f"   [ERRORE CRITICO] Impossibile connettersi a {w_name}: {conn_err}")
                    with results_lock:
                        if w_name in active_worker_names:
                            active_worker_names.remove(w_name)
                finally:
                    if worker_conn:
                        with self.connessioni_lock:
                            if worker_conn in self.connessioni_attive:
                                self.connessioni_attive.remove(worker_conn)
                        try:
                            worker_conn.close()
                        except Exception:
                            pass

            # 6. Avvio dei thread
            threads = []
            for name in worker_names:
                t = threading.Thread(target=worker_thread_consumer, args=(name,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # 7. Monitoraggio fallimento totale dello step
            if not task_queue.empty() and len(active_worker_names) == 0:
                print(f"   [{self.orchestrator_name}] Tutti i worker sono crashati. SQS gestirà il failover macro.")
                raise RuntimeError("Sotto-sistema Fault Tolerance interrotto: Nessun worker disponibile rimasto.")

            # Chiusura pulita delle connessioni
            print(f"[*] Pulizia risorse: chiusura di {len(self.connessioni_attive)} connessioni RPyC residue...")
            with self.connessioni_lock:
                for conn in self.connessioni_attive:
                    try: conn.close()
                    except Exception: pass

        # 8. Ricomposizione della foresta globale
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Ricomposizione foresta globale conforme a Scikit-Learn...")
            try:
                n_features = all_trained_trees[0].n_features_in_

                if tree_type == "classifier":
                    global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                    # Deriviamo le classi reali dagli alberi già addestrati (ogni DecisionTree
                    # fittato conserva il proprio attributo classes_), invece di assumere
                    # staticamente un problema binario con etichette {0, 1}. Con classi diverse
                    # (es. {1, 2} o multi-classe) l'assunzione fissa avrebbe silenziosamente
                    # etichettato male le predizioni finali.
                    trees_with_classes = [t for t in all_trained_trees if hasattr(t, "classes_")]
                    if trees_with_classes:
                        detected_classes = np.unique(np.concatenate([np.asarray(t.classes_) for t in trees_with_classes]))
                    else:
                        print(f"   [{self.orchestrator_name}] [WARN] Nessun albero espone 'classes_'. Fallback su {{0, 1}}.")
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

                print(f"   [{self.orchestrator_name}] Modello Globale salvato con successo in '{model_path}'.")

                # ─── MODIFICA 3: Restituiamo la dimensione REALE degli alberi salvati ───
                return len(all_trained_trees)

            except Exception as e:
                print(f"   [ERRORE AGGREGAZIONE] Fallimento durante l'unione dei sotto-modelli: {e}")
                traceback.print_exc()
                return len(all_trained_trees)

        print(f"   [{self.orchestrator_name}] Nessun albero collezionato.")
        # ─── Ritorna 0 se non è stato possibile generare o caricare nulla ───
        return 0

    def _execute_inference_step(self, payload: dict) -> dict:
        """
        Esegue l'inferenza distribuita centralizzata in modalità Fault-Tolerant
        sfruttando una task queue concorrente per riallocare dinamicamente i blocchi in caso di crash.
        """
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Target" if tree_type == "regressor" else "Label"

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA CENTRALIZZATA FAULT-TOLERANT ===")
        inference_start_time = time.perf_counter()
        model_path = self._resolve_model_path(job_id)

        # 1. RISOLUZIONE DINAMICA FILE MODELLO (.pkl) E TESTING SET (.csv) IN BASE ALL'AMBIENTE
        if self.environment == "aws":
            self.test_data_path = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{job_id}.csv"
        else:
            self.test_data_path = f"./.local_storage/shared_test_{job_id}.csv"

        print(f"[{self.orchestrator_name}] [AUTO-RESOLVE] Modello: {model_path} | Test Data: {self.test_data_path}")

        # 2. CARICAMENTO DELLA FORESTA (MODELLO GLOBALE AGGREGATO)
        if not self.checkpoint_dao.exists(model_path):
            raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")
        print(f"[{self.orchestrator_name}] Caricamento della foresta globale da {model_path}...")
        global_model = self.checkpoint_dao.load(model_path)
        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta caricata. Numero totale di alberi: {total_trees}")

        # 3. CARICAMENTO E PREPARAZIONE DEL DATASET DI TEST TRAMITE DAO
        print(f"[{self.orchestrator_name}] Caricamento Testing Set persistito via DAO: {self.test_data_path}")
        dao = DatasetDAOFactory.get_dao(self.environment)
        test_df = dao.load_dataset(self.test_data_path)

        print(f"[{self.orchestrator_name}] Preparazione della matrice di test (Shape: {test_df.shape})...")
        actual_target = target_col if target_col in test_df.columns else ("Target" if "Target" in test_df.columns else "Label")
        if actual_target != target_col:
            print(f"[{self.orchestrator_name}] [WARN] Colonna target attesa '{target_col}' non trovata nel test set. "
            f"Uso '{actual_target}' come fallback.")
        X_test = test_df.drop(columns=[actual_target]).to_numpy(dtype=np.float64)
        y_test = test_df[actual_target].to_numpy()
        serialized_X_test = pickle.dumps(X_test)

        # 4. SCOPERTA WORKER E INIZIALIZZAZIONE STRUTTURE FAULT-TOLERANT
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker disponibile nel Service Registry per l'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        # Calcolo dinamico granulare della dimensione del chunk di alberi
        CHUNK_SIZE = int(np.ceil(total_trees /num_workers))
        print(f"[{self.orchestrator_name}] CHUNK_SIZE di inferenza impostata a {CHUNK_SIZE} alberi per task.")

        # Popolamento della coda thread-safe dei sotto-task di inferenza
        task_queue = queue.Queue()
        tree_start = 0
        task_id_counter = 0
        predictions_chunks = self._load_inference_checkpoint(job_id)  # Tentativo di ripristino da checkpoint
        already_done_ranges = {start for start, _ in predictions_chunks}
        results_lock = threading.Lock()

        active_worker_names = list(worker_names)
        self.chunk_sent_event.clear()   # <-- reset, così ogni run è pulita
        while tree_start < total_trees:
            tree_end = min(tree_start + CHUNK_SIZE, total_trees)
            if tree_start not in already_done_ranges:
                chunk_estimators = all_trees[tree_start:tree_end]
                serialized_chunk_trees = pickle.dumps(chunk_estimators)
                task_queue.put((task_id_counter, tree_start, tree_end, serialized_chunk_trees))
                task_id_counter += 1
            else:
                print(f"[SHORT-CIRCUIT] Chunk {tree_start}-{tree_end} già completato. Skip.")
            tree_start = tree_end

        # Strutture dati condivise protette da Lock per i thread consumatori
        MAX_RETRIES_PER_TASK = 3  # Numero massimo di tentativi per ogni sotto-task prima di considerarlo fallito
        task_retries = {}  # Dizionario per tracciare i tentativi per ogni task_id

        failed_tasks = set()
        # 5. DEFINIZIONE DEL CONSUMATORE CONCORRENTE PER L'INFERENZA VIA RPC
        def inference_worker_consumer(w_name):
            rounds_done = 0
            w_info = available_workers[w_name]
            worker_conn = None
            try:
                print(f" [RPC INF -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                worker_conn = rpyc.connect(
                    w_info["host"],
                    w_info["port"],
                    config={
                        'allow_pickle': True,
                        'sync_request_timeout': RPC_INFERENCE_SYNC_TIMEOUT_SECONDS,
                        'keepalive': True
                    }
                )
                with self.connessioni_lock:
                    self.connessioni_attive.append(worker_conn)

                while True:
                    try:
                        task_id, start_idx, end_idx, chunk_trees_bytes = task_queue.get(timeout=2)
                        rounds_done += 1
                    except queue.Empty:
                        break

                    quota_alberi = end_idx - start_idx
                    print(f"[{self.orchestrator_name}-InfThread] Assegnazione Task {task_id} ({quota_alberi} alberi: {start_idx}-{end_idx}) a {w_name}")
                    self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="PROCESSING")
                    try:
                        self.chunk_sent_event.set()

                        # Invocazione remota sul metodo esposto dal BaseWorker
                        raw_response = worker_conn.root.predict_subset_forest(
                            chunk_trees_bytes,
                            serialized_X_test
                        )
                        sub_predictions = pickle.loads(obtain(raw_response))

                        with results_lock:
                            # Tracciamo start_idx per poter riordinare sequenzialmente i blocchi alla fine
                            predictions_chunks.append((start_idx, sub_predictions))
                            inference_cp_path = self._get_inference_checkpoint_path(job_id)
                            try:
                               self.checkpoint_dao.save(inference_cp_path, predictions_chunks)
                               print(f"   [RPC INF <- {w_name}] [CHECKPOINT INFERENZA OK] Task {task_id} archiviato. Progressivo in RAM/Storage: {len(predictions_chunks)} chunk.")
                            except Exception as e_fs:
                                print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere i chunk di inferenza parziali su file: {e_fs}")

                        print(f"   [RPC INF <- {w_name}] Task {task_id} completato con successo.")
                        self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="COMPLETED")
                        task_queue.task_done()

                    except Exception as e:
                        print(f"   [ERRORE RPC INFERENZA] Fallimento del worker {w_name} sul Task {task_id}: {e}")
                        retries = task_retries.get(task_id, 0) + 1
                        task_retries[task_id] = retries
                        if retries > MAX_RETRIES_PER_TASK:
                            # Segnaliamo il fallimento permanente invece di loopar all'infinito
                            print(f"[FATAL] Task {task_id} ha superato il limite di {MAX_RETRIES_PER_TASK} retry. Abort.")
                            self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="FAILED")
                            failed_tasks.add(task_id)
                            task_queue.task_done()
                        else:
                            # FAILOVER: Inserimento immediato del task interrotto nuovamente in coda
                            self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="REQUEUED")
                            task_queue.put((task_id, start_idx, end_idx, chunk_trees_bytes))
                            print(f"[{self.orchestrator_name}-InfThread] Task {task_id} riaccodato per il failover.")

                        with results_lock:
                            if w_name in active_worker_names:
                                active_worker_names.remove(w_name)
                        break  # Interruzione del loop per questo canale RPC corrotto
                print(f"[{w_name}] ha completato {rounds_done} round")
            except Exception as conn_err:
                print(f"   [ERRORE CONNESSIOINE INFERENZA] Impossibile raggiungere il worker {w_name}: {conn_err}")
                with results_lock:
                    if w_name in active_worker_names:
                        active_worker_names.remove(w_name)
            finally:
                if worker_conn:
                    with self.connessioni_lock:
                        if worker_conn in self.connessioni_attive:
                            self.connessioni_attive.remove(worker_conn)
                    try:
                        worker_conn.close()
                    except Exception:
                        pass

        # 6. AVVIO MULTI-THREADING E SINCRONIZZAZIONE DEI CONSUMATORI
        rpc_start_time = time.perf_counter()
        threads = []
        for name in worker_names:
            t = threading.Thread(target=inference_worker_consumer, args=(name,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        try:
            if failed_tasks:
                raise RuntimeError(f"Inferenza parziale: {len(failed_tasks)} chunk non completati.")
            if not task_queue.empty():
                raise RuntimeError("Task in coda orfani: tutti i worker sono crashati.")
        finally:
            with self.connessioni_lock:
                for conn in self.connessioni_attive:
                    try: conn.close()
                    except Exception: pass

        rpc_inference_time = time.perf_counter() - rpc_start_time

        # 7. ORDINAMENTO SEQUENZIALE E COMPOSIZIONE DELLA MATRICE DELLE PREDIZIONI
        print(f"[{self.orchestrator_name}] Collezionamento predizioni completato. Ricomposizione matrice in corso...")
        predictions_chunks.sort(key=lambda x: x[0])

        all_worker_predictions = []
        for _, sub_preds in predictions_chunks:
            all_worker_predictions.extend(sub_preds)

        predictions_matrix = np.array(all_worker_predictions)
        print(f"[{self.orchestrator_name}] Matrice complessiva delle predizioni rigenerata: {predictions_matrix.shape}")

        total_inference_time = time.perf_counter() - inference_start_time

        # 8. AGGREGAZIONE DEL VOTO DI MAGGIORANZA (CLASSIFICAZIONE) / MEDIA (REGRESSIONE)
        # FRA GLI ALBERI, seguita dal calcolo delle metriche sulla predizione finale.
        final_predictions, y_probs = self._aggregate_forest_predictions(
            predictions_matrix=predictions_matrix,
            tree_type=tree_type
        )
        metrics = self.calculate_metrics(
            final_predictions=final_predictions,
            y_test=y_test,
            tree_type=tree_type,
            y_probs=y_probs
        )
        self._save_metrics(job_id, "inference", {
            "job_id": job_id, "mode": "centralized", "phase": "inference",
            "tree_type": tree_type, "testing_set_size": X_test.shape[0],
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

        # Esposto esplicitamente (prima mancava): chi chiama questo metodo — inclusa
        # la suite di test locale — deve poter leggere le metriche reali dal valore di
        # ritorno, invece di affidarsi a un monkey-patch interno fragile.
        return {
            "status": "SUCCESS" if not failed_tasks else "PARTIAL",
            "testing_set_size": int(X_test.shape[0]),
            "total_inference_time": total_inference_time,
            "rpc_inference_time": rpc_inference_time,
            "metrics": metrics
        }

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int, alberi_reali: list = None):
        """
        Estende il checkpoint della classe base aggiungendo il salvataggio FISICO
        degli alberi (specifico del calcolo centralizzato).
        """
        # 1. Chiamiamo la classe base per aggiornare DynamoDB (evita duplicazione di codice)
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

        # 2. Se ci sono alberi fisici da blindare su disco/S3, lo facciamo qui
        if alberi_reali is not None and len(alberi_reali) > 0:
            checkpoint_trees_path = self._resolve_trees_checkpoint_path(job_id)
            try:
                self.checkpoint_dao.save(checkpoint_trees_path, alberi_reali)
                print(f"[{self.orchestrator_name}] [CENTRALIZED-CHECKPOINT-FISICO] {len(alberi_reali)} alberi salvati in storage.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE STORAGE] Fallito salvataggio fisico degli alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        """
        Override del metodo di pulizia per rimuovere il file pickle parziale.
        """
        super()._clean_checkpoint(job_id)
        self._trees_cache.pop(job_id, None)
        checkpoint_trees_path = self._resolve_trees_checkpoint_path(job_id)
        try:
            self.checkpoint_dao.delete(checkpoint_trees_path)
            print(f"[{self.orchestrator_name}] [CLEAN OK] Rimosso checkpoint degli alberi parziali.")
        except Exception as e:
            print(f"[{self.orchestrator_name}] [CLEAN WARN] Impossibile cancellare {checkpoint_trees_path}: {e}")

        inference_cp = self._get_inference_checkpoint_path(job_id)
        try:
            self.checkpoint_dao.delete(inference_cp)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [CLEAN WARN] Impossibile cancellare {inference_cp}: {e}")

    def _resolve_trees_checkpoint_path(self, job_id: str) -> str:

        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/checkpoints/checkpoint_trees_{job_id}.pkl"
        return f"./.local_storage/checkpoint_trees_{job_id}.pkl"

    def _resolve_model_path(self, job_id: str) -> str:
        """Path del modello globale aggregato, in una sotto-cartella dedicata alla
        modalità centralizzata per evitare collisioni col modello federato in caso
        di job_id riutilizzati tra le due modalità."""
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/centralized/model_{job_id}.pkl"
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

if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Centralizzato...")
    orchestrator = CentralizedOrchestrator()
    orchestrator.start()