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
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator, env_timeout_seconds
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.featureselection import CICIDSFeatureSelector
from src.dataset.checkpoint_dao import CheckpointDAOFactory
from src.shared.utilities.task_storage import (
    load_task_from_shared_storage,
    save_bytes_to_shared_storage,
    load_bytes_from_shared_storage,
)


TEST_SIZE = 0.2
BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")

# Timeout (in secondi) delle chiamate RPC sincrone verso i worker.
#
# PRIMA: due letterali 600 incastonati nelle chiamate a rpyc.connect (nel thread
# di dispatch dell'addestramento e in quello dell'inferenza). deploy.sh leggeva
# RPC_SYNC_TIMEOUT_SECONDS / RPC_INFERENCE_SYNC_TIMEOUT_SECONDS dal .env e le
# iniettava nella task definition ECS dell'orchestratore, ma il codice
# centralizzato non le leggeva: la configurazione c'era, era documentata, e non
# aveva alcun effetto. Solo federated.py le usava davvero.
#
# I DEFAULT RESTANO 600/600, non i 1800/900 di federated.py: così, quando le
# variabili non sono impostate — cioè in locale e in Docker Compose — il
# comportamento è identico byte per byte a quello precedente. Su AWS, dove
# deploy.sh le valorizza, il timeout diventa finalmente quello dichiarato nel
# .env, che è il punto di tutta questa configurazione.
RPC_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_SYNC_TIMEOUT_SECONDS", 600)
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", 600)

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, orchestrator_name: str = None):
        self.cfg = SystemConfig()
        name = orchestrator_name or f"Orchestrator-Centralizzato-{socket.gethostname()}"

        self.current_job_id = None
        self.train_data_path = None
        self.test_data_path = None
        self.chunk_sent_event = threading.Event()
        self._trees_cache = {}
        # Durata dell'ultima fase di preparazione dati (ETL). Serve agli scenari
        # di test per scomporre il tempo totale in "preparazione dati" +
        # "addestramento distribuito": la baseline locale misura t_seq sul solo
        # fit, quindi confrontarla con un totale che include l'ETL (30-40s su
        # AWS per via di S3) penalizzerebbe sistematicamente il cluster.
        # 0.0 quando l'ETL viene saltata grazie allo SHORT-CIRCUIT.
        self.last_etl_seconds = 0.0
        # Scomposizione del tempo di _execute_training_step, esposta perché il
        # confronto con la baseline locale sia onesto in entrambe le direzioni.
        # La baseline misura il solo fit di scikit-learn: sommarci sopra
        # trasferimenti S3, checkpoint e stima OOB — che la baseline non fa
        # affatto — penalizzerebbe il cluster per lavoro che non gli è stato
        # chiesto di confrontare.
        #
        #   last_dispatch_seconds     costruzione vera degli alberi: scoperta
        #                             dei worker, invio dei chunk via RPC e
        #                             attesa del loro completamento. È IL
        #                             numero da confrontare con T_seq/T_1node.
        #   last_aggregation_seconds  ricomposizione della foresta globale e
        #                             salvataggio del modello sullo storage.
        #   last_oob_seconds          stima Out-Of-Bag: ricarica il training
        #                             set e ricalcola le predizioni, quindi è
        #                             una diagnostica aggiuntiva, non parte
        #                             dell'addestramento.
        #
        # Totale di _execute_training_step ~=
        #   last_etl_seconds + last_dispatch_seconds
        #   + last_aggregation_seconds + last_oob_seconds
        self.last_dispatch_seconds = 0.0
        self.last_aggregation_seconds = 0.0
        self.last_oob_seconds = 0.0
        
        super().__init__(
            orchestrator_name=name,
            queue_name=self.cfg.sqs_centralized_queue
        )
        self.checkpoint_dao = CheckpointDAOFactory.get_dao(self.environment)

    def _load_training_matrix_for_oob(self, tree_type: str):
        """
        Ricarica il training set condiviso (self.train_data_path) riproducendo
        ESATTAMENTE la stessa risoluzione della colonna target e lo stesso casting
        usati da CentralizedWorker._load_data: è essenziale che l'ordine delle
        righe risultante coincida con quello visto dai worker in fase di training,
        perché gli indici OOB salvati su ogni albero sono posizionali rispetto a
        QUELLA matrice.
        """
        dao = DatasetDAOFactory.get_dao(self.environment)
        df = dao.load_dataset(self.train_data_path)

        target_column = "Target" if tree_type == "regressor" else "Label"
        actual_target = target_column if target_column in df.columns else (
            "Target" if "Target" in df.columns else "Label"
        )
        feature_cols = [c for c in df.columns if c != actual_target]

        X = df[feature_cols].to_numpy(dtype=np.float64)
        y_df = df[actual_target]
        if tree_type == "regressor":
            y = y_df.to_numpy(dtype=np.float64)
        else:
            y = y_df.to_numpy(dtype=np.int64)
        return X, y

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
            # Letto dal manifesto (scritto da run_baseline.py) invece di un
            # letterale fisso: permette di rilanciare lo stesso job con soglie
            # diverse (es. 0.0 per disattivare il filtro di correlazione e
            # tenere solo la rimozione delle feature a varianza zero) senza
            # dover editare questo file — utile per un ablation study.
            correlation_threshold = payload.get("correlation_threshold", 0.05)
            fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=correlation_threshold)
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
            self.last_etl_seconds = time.perf_counter() - t0
            print(f"[DEBUG TIMING] _prepare_data completato in {self.last_etl_seconds:.2f}s")
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
                self.last_etl_seconds = 0.0
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
        if start_alberi == 0:
            # Rimuove parti incrementali E monolitico: ripartendo da zero non
            # deve sopravvivere nulla di un tentativo precedente sullo stesso id.
            self._purge_trees_checkpoint(self.current_job_id)
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
                if self._trees_checkpoint_exists(self.current_job_id):
                    try:
                        # Ricompone dalle parti incrementali, con fallback
                        # automatico sul formato monolitico precedente.
                        all_trained_trees = self._load_trees_checkpoint(self.current_job_id)

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
        min_samples_split = hp.get("min_samples_split", 2)
        # class_weight ha senso solo in classificazione: il worker lo ignora comunque
        # per i regressori, ma evitiamo di forzarlo se il payload non lo prevede.
        class_weight = hp.get("class_weight", None)
        criterion = hp.get("criterion", None)
        # Inoltrati esplicitamente al worker: prima non venivano trasmessi
        # affatto e ogni albero usava i valori di boot del worker
        # (self.bootstrap / self.max_samples), rendendo di fatto inerte quanto
        # dichiarato nel manifesto della baseline. None = "non specificato",
        # e il worker mantiene i propri valori di boot (comportamento storico).
        bootstrap = hp.get("bootstrap", None)
        max_samples = hp.get("max_samples", None)
        print(f"[{self.orchestrator_name}] Iperparametri effettivi -> n_estimators(step)={total_step_trees}, "
              f"max_depth={max_depth}, max_features={max_features}, min_samples_split={min_samples_split}, "
              f"criterion={criterion}, bootstrap={bootstrap}, max_samples={max_samples}")

        # Azzerati a ogni step: se questo step non costruisce alberi (caso
        # limite sotto) i valori devono restare 0.0 e non conservare quelli
        # dello step precedente.
        self.last_dispatch_seconds = 0.0
        self.last_aggregation_seconds = 0.0
        self.last_oob_seconds = 0.0

        # Caso limite: già finito tutto ma eravamo crashati prima di consolidare
        if total_step_trees <= 0:
            print(f"[{self.orchestrator_name}] Tutti gli alberi richiesti ({len(all_trained_trees)}) sono già pronti in memoria.")
        else:
            print(f"\n [{self.orchestrator_name}] Distribuzione carico residuo: {total_step_trees} alberi da generare...")
            dispatch_start = time.perf_counter()
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

            # Lock DEDICATO alla persistenza del checkpoint, separato da
            # results_lock. Prima l'upload su S3 avveniva dentro results_lock,
            # cioè dentro la stessa sezione critica che serve ad accodare gli
            # alberi ricevuti: ogni worker che finiva restava fermo ad aspettare
            # la fine dell'upload di un altro, non per calcolare ma solo per
            # poter registrare il proprio risultato. Era un punto di
            # serializzazione che cresceva col numero di worker, e falsava
            # proprio la misura di strong scaling.
            checkpoint_lock = threading.Lock()
            # Contatore monotono dell'ultimo snapshot effettivamente persistito.
            # Serve a due cose:
            #  1) impedire che uno snapshot più VECCHIO sovrascriva uno più
            #     recente — ora che la scrittura è fuori da results_lock, due
            #     thread possono arrivarci in ordine diverso da quello in cui
            #     hanno preso lo snapshot, e un checkpoint che regredisce
            #     sposterebbe INDIETRO il punto di ripartenza dopo un guasto;
            #  2) saltare le scritture già superate. Se quando un thread ottiene
            #     il lock risulta già persistito uno snapshot con più alberi, il
            #     suo è ridondante: il checkpoint resta comunque più avanti, la
            #     tolleranza ai guasti non peggiora e si risparmiano byte.
            # "parts" riparte dal numero di parti gia'su storage: scrivere di nuovo
            # dalla 0 sovrascriverebbe un delta valido con un altro delta.
            last_checkpointed = {"count": start_alberi,
                                 "parts": self._count_trees_checkpoint_parts(self.current_job_id)}

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

                            ack_raw = worker_conn.root.train_subset_forest(
                                source_info=source_info,
                                num_trees=quota_chunk,       
                                base_seed=chunk_seed,    
                                max_depth=max_depth,
                                tree_type=hp.get("tree_type"),
                                max_features=max_features,
                                min_samples_split=min_samples_split,
                                class_weight=class_weight,
                                criterion=criterion,
                                bootstrap=bootstrap,
                                max_samples=max_samples
                            )

                            # Il worker NON restituisce più il blob degli alberi
                            # (fino a 1+ GB su scenari di scalabilità) come valore
                            # di ritorno RPC: lo ha già persistito nello storage
                            # condiviso (S3/locale) prima di rispondere, e qui ci
                            # limitiamo a un piccolo ack + rilettura diretta dallo
                            # storage. Evita l'hang osservato quando RPyC deve
                            # trasportare un payload sincrono molto grande come
                            # valore di ritorno (vedi Scenario 2 - Scalabilità).
                            ack = obtain(ack_raw)
                            if not isinstance(ack, dict) or not ack.get("ack"):
                                raise RuntimeError(
                                    f"Risposta inattesa dal worker {w_name} per il task {task_id}: {ack!r}"
                                )

                            result_trees_bytes = load_task_from_shared_storage(
                                source_info, chunk_seed, quota_chunk,
                                self.environment, self.orchestrator_name
                            )
                            if result_trees_bytes is None:
                                raise RuntimeError(
                                    f"Worker {w_name}: task {task_id} confermato (ack) ma il blob "
                                    f"non è stato trovato nello storage condiviso."
                                )
                            result_trees = pickle.loads(result_trees_bytes)
                            
                            # SEZIONE CRITICA MINIMA: solo l'aggiornamento della
                            # lista condivisa e uno snapshot immutabile. L'upload
                            # su S3 e la scrittura su DynamoDB, che prima stavano
                            # qui dentro, sono stati spostati FUORI: tenerli nel
                            # lock significava che ogni worker che finiva restava
                            # bloccato dietro l'upload di un altro solo per poter
                            # registrare il proprio risultato.
                            with results_lock:
                                all_trained_trees.extend(result_trees)
                                current_total = len(all_trained_trees)
                                # list(...) crea una copia: la serializzazione fuori
                                # dal lock non deve poter vedere la lista mutare.
                                snapshot = list(all_trained_trees)

                            # --- fuori da results_lock ---
                            with checkpoint_lock:
                                if current_total > last_checkpointed["count"]:
                                    try:
                                        # Scrive SOLO gli alberi nuovi (alla parte 0
                                        # l'intero snapshot, per migrare dal formato
                                        # monolitico). Traffico totale: N invece di N*(W+1)/2.
                                        self._persist_trees_delta(
                                            self.current_job_id, snapshot,
                                            last_checkpointed["count"], last_checkpointed["parts"])
                                        last_checkpointed["count"] = current_total
                                        last_checkpointed["parts"] += 1
                                        # La cache di istanza viene allineata SOLO dopo che il
                                        # salvataggio fisico è andato a buon fine: così non è mai
                                        # "più avanti" della fonte di verità persistita, che è
                                        # ciò che un'altra istanza rileggerebbe in caso di failover.
                                        self._trees_cache[self.current_job_id] = snapshot
                                        print(f"   [RPC <- {w_name}] [CHECKPOINT FS OK] Task {task_id} archiviato. Progressivo in RAM/Storage: {current_total} alberi.")
                                    except Exception as e_fs:
                                        # last_checkpointed NON avanza: un writer successivo
                                        # deve poter riprovare a persistere lo stato.
                                        print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere gli alberi parziali su file: {e_fs}")

                                    # Il contatore logico segue lo stesso ordine monotono del
                                    # checkpoint fisico, così i due non possono divergere.
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
                                else:
                                    # Snapshot superato: sullo storage c'è già uno stato con
                                    # PIÙ alberi, quindi riscriverlo non aggiungerebbe nulla e
                                    # anzi farebbe REGREDIRE il punto di ripartenza.
                                    print(f"   [RPC <- {w_name}] [CHECKPOINT SKIP] Task {task_id}: già persistito uno "
                                          f"stato più avanzato ({last_checkpointed['count']} alberi >= {current_total}).")

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

            # Fine della fase di costruzione degli alberi: da qui in poi è solo
            # ricomposizione e diagnostica. Questo è il tempo direttamente
            # confrontabile con T_seq/T_1node della baseline locale.
            self.last_dispatch_seconds = time.perf_counter() - dispatch_start
            print(f"[DEBUG TIMING] Costruzione distribuita degli alberi completata in "
                  f"{self.last_dispatch_seconds:.2f}s ({len(all_trained_trees)} alberi, "
                  f"{num_workers} worker).")

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
            aggregation_start = time.perf_counter()
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

                # La ricomposizione e il salvataggio finiscono qui: la stima OOB
                # che segue è una diagnostica separata (ricarica il training set
                # e ricalcola le predizioni), quindi va cronometrata a parte per
                # non gonfiare il costo attribuito all'addestramento.
                self.last_aggregation_seconds = time.perf_counter() - aggregation_start
                print(f"[DEBUG TIMING] Ricomposizione e salvataggio del modello: "
                      f"{self.last_aggregation_seconds:.2f}s.")

                # ─── STIMA OOB (Breiman, 2001), "gratis" e non bloccante ───
                # Se fallisce per qualunque motivo, non deve invalidare un training
                # già completato e salvato con successo: solo log, nessun raise.
                oob_start = time.perf_counter()
                try:
                    X_train_oob, y_train_oob = self._load_training_matrix_for_oob(tree_type)
                    oob_metrics = self._compute_oob_metrics(
                        all_trees=all_trained_trees,
                        X_train=X_train_oob,
                        y_train=y_train_oob,
                        tree_type=tree_type
                    )
                    if oob_metrics is not None:
                        self._save_metrics(self.current_job_id, "training_oob", {
                            "job_id": self.current_job_id, "mode": "centralized", "phase": "training_oob",
                            "tree_type": tree_type, "n_estimators": len(all_trained_trees),
                            "metrics": oob_metrics
                        })
                except Exception as e_oob:
                    print(f"   [{self.orchestrator_name}] [OOB-WARN] Stima OOB fallita (training non impattato): {e_oob}")
                finally:
                    # Cronometrata anche in caso di fallimento: se l'OOB si
                    # interrompe a metà, il tempo speso è comunque reale e non
                    # va attribuito silenziosamente all'addestramento.
                    self.last_oob_seconds = time.perf_counter() - oob_start
                    print(f"[DEBUG TIMING] Stima OOB (ricarica training set + predizioni): "
                          f"{self.last_oob_seconds:.2f}s.")

                print(f"[DEBUG TIMING] Riepilogo _execute_training_step -> "
                      f"ETL {self.last_etl_seconds:.2f}s | costruzione alberi "
                      f"{self.last_dispatch_seconds:.2f}s | aggregazione "
                      f"{self.last_aggregation_seconds:.2f}s | OOB {self.last_oob_seconds:.2f}s")

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

        # Spazio di classi GLOBALE (calcolato in fase di training su TUTTI gli alberi):
        # serve ai worker per allineare le colonne di predict_proba di ogni singolo
        # albero, anche quando un albero non ha visto tutte le classi nel proprio
        # campione bootstrap.
        global_classes = global_model.classes_.tolist() if tree_type == "classifier" else None

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
                # Il chunk di alberi NON viaggia più come argomento della RPC
                # (fino a 1+ GB con pochi worker attivi, stesso problema già
                # risolto in training - vedi hang/timeout SSM osservato in
                # Scenario 2): lo carichiamo una volta sullo storage condiviso
                # e passiamo al worker solo la chiave, che lo riscarica da sé.
                chunk_key = f"inference_chunks/{job_id}/chunk_{tree_start}_{tree_end}.pkl"
                save_bytes_to_shared_storage(chunk_key, serialized_chunk_trees, self.environment, self.orchestrator_name)
                task_queue.put((task_id_counter, tree_start, tree_end, chunk_key))
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
                        task_id, start_idx, end_idx, chunk_key = task_queue.get(timeout=2)
                        rounds_done += 1
                    except queue.Empty:
                        break

                    quota_alberi = end_idx - start_idx
                    print(f"[{self.orchestrator_name}-InfThread] Assegnazione Task {task_id} ({quota_alberi} alberi: {start_idx}-{end_idx}) a {w_name}")
                    self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="PROCESSING")
                    try:
                        self.chunk_sent_event.set()
                        
                        # Invocazione remota sul metodo esposto dal BaseWorker:
                        # 'chunk_key' è solo il riferimento allo storage condiviso,
                        # non il blob — il worker lo scarica direttamente da lì.
                        raw_response = worker_conn.root.predict_subset_forest(
                            chunk_key, 
                            serialized_X_test,
                            tree_type,
                            global_classes
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
                            task_queue.put((task_id, start_idx, end_idx, chunk_key))
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

        # 8. AGGREGAZIONE: SOFT VOTING (media delle probabilità per-albero + argmax) per
        # la classificazione, MEDIA per la regressione, seguita dal calcolo delle metriche
        # sulla predizione finale.
        final_predictions, y_probs = self._aggregate_forest_predictions(
            predictions_matrix=predictions_matrix,
            tree_type=tree_type,
            global_classes=global_classes
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
            try:
                # Sostituzione integrale dello stato: si azzera e si riscrive come
                # parte 0. Percorso oggi mai esercitato — BaseOrchestrator chiama
                # _save_checkpoint senza 'alberi_reali' — ma va tenuto coerente
                # col formato a parti, altrimenti reintrodurrebbe un monolitico.
                self._purge_trees_checkpoint(job_id)
                self._persist_trees_delta(job_id, alberi_reali, 0, 0)
                print(f"[{self.orchestrator_name}] [CENTRALIZED-CHECKPOINT-FISICO] {len(alberi_reali)} alberi salvati in storage.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE STORAGE] Fallito salvataggio fisico degli alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        """
        Override del metodo di pulizia per rimuovere il file pickle parziale.
        """
        super()._clean_checkpoint(job_id)
        self._trees_cache.pop(job_id, None)
        # Rimuove tutte le parti incrementali oltre all'eventuale monolitico.
        try:
            self._purge_trees_checkpoint(job_id)
            print(f"[{self.orchestrator_name}] [CLEAN OK] Rimosso checkpoint degli alberi parziali.")
        except Exception as e:
            print(f"[{self.orchestrator_name}] [CLEAN WARN] Impossibile cancellare il checkpoint alberi: {e}")
 
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