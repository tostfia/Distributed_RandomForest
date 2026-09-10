import pickle
import os
import gc
import random
import socket
import threading
import time
import traceback
import json
from botocore.exceptions import ClientError
import boto3
import rpyc
import numpy as np
import re

from rpyc.utils.classic import obtain
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_recall_curve, roc_auc_score
from src.dataset.checkpoint_dao import CheckpointDAOFactory
from src.shared.utilities.task_storage import load_task_trees_from_shared_storage
from src.orchestrator.BaseOrchestrator import BaseOrchestrator, env_timeout_seconds
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.config import SystemConfig

BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")

# Timeout (in secondi) per le chiamate RPC sincrone verso i worker. Configurabile via
# .env / variabile d'ambiente per poter alzarlo su AWS (dove un singolo worker può
# dover addestrare/predire un chunk molto più grande che in locale, es. scenario di
# scalabilita' con pochi worker attivi), senza toccare il default usato finora in
# locale/Docker se la variabile non e' impostata. Due costanti separate perché
# training e inferenza avevano gia' default diversi (600s e 300s).
#
# int(os.environ.get(...)) è stato sostituito da env_timeout_seconds: Terraform (ecs_task_definitions.tf),
# quando la chiave manca nel .env, ripiega su "1800s"/"900s" — col suffisso — e
# int("1800s") solleva ValueError a livello di modulo, uccidendo il container
# all'import. Vedi il docstring di env_timeout_seconds in BaseOrchestrator.py.
RPC_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_SYNC_TIMEOUT_SECONDS", 1800)
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", 900)

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
        self.worker_wait_timeout = float(os.environ.get("FED_WORKER_WAIT_TIMEOUT_SECONDS", 60))

        # Cache in-memoria (solo per QUESTA istanza di processo) degli alberi
        # già addestrati per un dato job. Serve esclusivamente a evitare una
        # GET S3 ridondante quando il round successivo viene gestito dalla
        # STESSA istanza orchestratore. NON sostituisce mai il checkpoint
        # fisico su S3, che resta l'unica fonte di verità condivisa: se
        # un'altra istanza (nuovo leader dopo un fault) subentra, questa
        # cache sarà vuota/non coerente e si procederà comunque con un
        # reload reale da S3 (vero FAILOVER-RESUME), garantendo il failover.
        self._trees_cache = {}

        # Strumentazione dei tempi (letta da performance.py/scalability.py via
        # getattr): prima assente su FederatedOrchestrator, che ha la propria
        # implementazione di _execute_training_step separata da quella
        # centralizzata dove questi attributi erano già impostati. Inizializzati
        # qui (non solo assegnati a runtime) così un getattr(...) prima del
        # primo job trova comunque 0.0 invece di ricadere sul default silenzioso
        # del chiamante, che mascherava l'assenza di dato reale.
        self.last_etl_seconds = 0.0
        self.last_dispatch_seconds = 0.0
        self.last_aggregation_seconds = 0.0
        # Il federato non esegue stima OOB (scelta di design: l'OOB richiede il
        # training set completo in un unico posto, qui ogni worker vede solo il
        # proprio shard) — resta sempre 0.0, non "non misurato".
        self.last_oob_seconds = 0.0

    def _ensure_local_bootstrap(self, payload: dict):
        """
        VERIFICA (senza generarli) che gli shard federati siano già presenti sul
        filesystem locale per l'ambiente 'local'. La generazione/sharding NON
        avviene più qui: è responsabilità di uno script di provisioning standalone
        (script_local/provision_local_shards.py), eseguito UNA VOLTA, PRIMA di
        avviare master e worker — coerente con quanto già fa _ensure_aws_bootstrap
        per l'ambiente AWS, e con l'idea che in uno scenario federato i dati
        risiedano già sui nodi quando il sistema parte, invece di essere
        generati/distribuiti reattivamente durante un job.

        Per il dataset 'synthetic' non serve alcun provisioning: ogni worker
        genera autonomamente il proprio shard sintetico al boot.
        """
        if self.environment != "local":
            print(f"[{self.orchestrator_name}] Ambiente Cloud/AWS rilevato. Bootstrap locale saltato.")
            return
        datasetype = self._resolve_dataset_type(payload)
        if datasetype == "synthetic":
            print(f"[{self.orchestrator_name}] Dataset SINTETICO rilevato. Nessun controllo shard necessario "
                  f"(generato autonomamente da ogni worker).")
            return

        num_workers = self.num_workers
        base_cache_dir = "./workers_cache"
        print(f"[{self.orchestrator_name}] [CHECK LOCAL] Verifica provisioning shard su disco per {num_workers} worker...")

        mancanti = []
        for i in range(1, num_workers + 1):
            dir_padded = os.path.join(base_cache_dir, f"Worker-Locale-{i:02d}")
            dir_unpadded = os.path.join(base_cache_dir, f"Worker-Locale-{i}")

            train_p = os.path.join(dir_padded, "train_shard.csv")
            test_p = os.path.join(dir_padded, "test_shard.csv")
            train_up = os.path.join(dir_unpadded, "train_shard.csv")
            test_up = os.path.join(dir_unpadded, "test_shard.csv")

            if not ((os.path.exists(train_p) and os.path.exists(test_p)) or
                    (os.path.exists(train_up) and os.path.exists(test_up))):
                mancanti.append(f"Worker-Locale-{i:02d}")

        if mancanti:
            raise RuntimeError(
                f"[{self.orchestrator_name}] Provisioning locale incompleto: mancano gli shard per "
                f"{len(mancanti)} worker (in '{base_cache_dir}'), es. {mancanti[:3]}. Esegui "
                f"'python -m script_local.provision_local_shards --num-workers {num_workers}' "
                f"prima di avviare il cluster."
            )
        print(f"[{self.orchestrator_name}] [CHECK LOCAL OK] Tutti gli shard richiesti sono presenti su disco.")

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

    def _read_synthetic_total_n_samples(self) -> int:
        """
        Legge 'n_samples' (il TOTALE concettuale del dataset sintetico, non
        la quota di un singolo worker) da outputs_baseline/config_synthetic.json
        -- stesso file/pattern di lettura di
        BaseOrchestrator.read_decision_threshold_from_config, qui riusato per
        lo stesso scopo: allineare il federato allo stesso "spazio
        campionario" del centralizzato, invece di un fallback fisso
        indipendente dal numero di worker (vedi _load_synthetic_data in
        federatedWorker.py, che genera 166666 campioni per worker a
        prescindere da quanti worker sono attivi -- coincidenza numerica con
        1.000.000/6, non un calcolo dinamico: con un numero di worker diverso
        da 6 il dataset "federato" smetteva di corrispondere, come TOTALE
        aggregato, a quello centralizzato).

        Ritorna None (mai un'eccezione) se il file non esiste, non contiene
        'n_samples', o è malformato: in quel caso il chiamante deve ricadere
        sul comportamento precedente (nessun override esplicito, il worker
        usa il proprio fallback fisso).
        """
        config_filename = "config_synthetic.json"
        config_path = os.path.join(os.getcwd(), "outputs_baseline", config_filename)

        if not os.path.exists(config_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
            config_path = os.path.join(project_root, "outputs_baseline", config_filename)

        if not os.path.exists(config_path):
            print(f"[{self.orchestrator_name}] [ATTENZIONE] {config_filename} non trovato: "
                  f"nessun totale campioni sintetici da ripartire tra i worker, ogni worker "
                  f"userà il proprio fallback fisso.")
            return None

        try:
            with open(config_path, "r") as f:
                config_dati = json.load(f)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE] Lettura {config_filename} fallita: {e}")
            return None

        total_n_samples = config_dati.get("n_samples")
        if total_n_samples is None:
            print(f"[{self.orchestrator_name}] [ATTENZIONE] 'n_samples' assente in "
                  f"{config_filename}: ogni worker userà il proprio fallback fisso.")
            return None

        return int(total_n_samples)

    def _allocate_synthetic_samples(self, worker_names: list, total_n_samples: int) -> dict:
        """
        Ripartisce 'total_n_samples' in quote il più possibile uguali tra i
        worker forniti (divisione intera + resto distribuito ai primi worker
        nell'ordine di 'worker_names', così la somma delle quote torna
        ESATTAMENTE 'total_n_samples' indipendentemente da quanti worker sono
        attivi -- a differenza del fallback fisso precedente, la dimensione
        totale del dataset sintetico federato ora corrisponde sempre a quella
        del centralizzato, qualunque sia NUM_WORKERS.
        """
        num_workers = len(worker_names)
        base_quota, remainder = divmod(total_n_samples, num_workers)
        quotas = {}
        for i, w_name in enumerate(worker_names):
            quotas[w_name] = base_quota + (1 if i < remainder else 0)
        return quotas

    def _fetch_worker_shard_sizes(self, worker_names: list, available_workers: dict, job_id: str = None) -> dict:
        """
        Interroga in parallelo ogni worker per la dimensione del proprio shard di
        training locale (exposed_get_local_shard_size), PRIMA di allocare il
        budget di alberi del round: è il prerequisito per una ripartizione
        proporzionale alla quantità di dati posseduti, invece che equa a
        prescindere (fondamentale con partizionamento non-IID). Timeout breve
        per singola chiamata: è solo un conteggio righe su un file già locale,
        non deve competere in durata con le RPC di training vere e proprie.
        Un worker che non risponde in tempo viene semplicemente OMESSO dal
        dizionario risultato (non con size=0): la gestione del fallback per i
        worker "senza dimensione nota" è delegata a _allocate_tree_quotas.

        job_id: inoltrato a exposed_get_local_shard_size così un worker che ha
        già preprocessato lo shard per QUESTO job (round >= 2 dello stesso
        job) può riportare il conteggio reale post-undersampling invece
        della sola stima sul CSV grezzo — vedi quel metodo per i dettagli.
        """
        sizes = {}
        lock = threading.Lock()

        def _probe(w_name):
            w_info = available_workers.get(w_name)
            if not w_info:
                return
            conn = None
            try:
                conn = rpyc.connect(
                    w_info["host"], w_info["port"],
                    config={"allow_pickle": True, "sync_request_timeout": 30}
                )
                size = int(obtain(conn.root.exposed_get_local_shard_size(job_id)))
                with lock:
                    sizes[w_name] = size
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Impossibile ottenere la dimensione dello "
                      f"shard da '{w_name}' ({e}): riceverà una quota di fallback.")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        threads = [threading.Thread(target=_probe, args=(w,)) for w in worker_names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=35)
        return sizes

    def _allocate_tree_quotas(self, total_step_trees: int, worker_names: list, worker_shard_sizes: dict,
                               strategy: str = "proportional") -> dict:
        """
        Alloca il budget di alberi da addestrare in QUESTO round tra i worker
        (metodo dei resti più grandi: garantisce che la somma delle quote sia
        ESATTAMENTE total_step_trees, senza doverla clampare a valle come
        nella vecchia ripartizione a CHUNK_SIZE fisso).

        strategy="proportional" (default): pesa ogni worker in base alla
        dimensione del proprio shard locale — corrisponde alla formula di
        FedAvg n_k/n applicata al numero di alberi anziché ai pesi del
        modello. Con partizionamento IID gli shard sono quasi uguali per
        costruzione, quindi il risultato è praticamente indistinguibile dalla
        ripartizione equa. Con partizionamento non-IID (dirichlet/by_day),
        dove le dimensioni possono differire di molto, evita di addestrare
        tanti alberi quanto un worker "ricco di dati" su uno shard minuscolo:
        alberi ad alta varianza che, nel soft voting finale, peserebbero
        comunque quanto tutti gli altri.

        strategy="equal": stessa quota a tutti i worker indipendentemente
        dalla dimensione dello shard (comportamento storico). Utile come
        confronto controllato: con partizionamento non-IID estremo, pesare
        per dimensione può ridurre la rappresentazione nella foresta finale
        di un worker che detiene un pattern raro ma prezioso (es. una classe
        minoritaria concentrata su quel worker) — "equal" garantisce a quel
        worker la stessa voce in capitolo degli altri, a scapito di dedicare
        più alberi a shard piccoli e ad alta varianza.

        Worker senza dimensione nota (RPC fallita, solo rilevante per
        "proportional") ricevono la dimensione MEDIA dei worker noti come
        stima di fallback, invece di 0 (che li escluderebbe implicitamente
        dal round). Se NESSUN worker ha una dimensione nota (es.
        dataset_type='synthetic', dove lo shard non esiste ancora sul disco a
        questo punto), si ricade comunque sulla ripartizione equa.
        """
        if strategy == "equal":
            sizes = {w: 1 for w in worker_names}
            print(f"[{self.orchestrator_name}] Allocazione alberi EQUA (tree_allocation_strategy='equal'): "
                  f"ogni worker riceve la stessa quota indipendentemente dalla dimensione del proprio shard.")
        else:
            sizes = dict(worker_shard_sizes)
            # IMPORTANTE: "sconosciuto" (RPC fallita) è diverso da "noto e pari a
            # zero" (shard genuinamente vuoto, es. Dirichlet con alpha estremo che
            # non assegna alcuna riga di quella classe a quel worker). Solo il
            # primo caso va coperto con una stima di fallback; il secondo va
            # rispettato così com'è — un worker con shard vuoto deve ricevere
            # quota 0, non una quota "media" che lo manderebbe in errore al primo
            # bootstrap su un array senza campioni.
            missing = [w for w in worker_names if w not in sizes]
            known = [w for w in worker_names if w in sizes]

            if missing and known and sum(sizes[w] for w in known) > 0:
                fallback_size = max(1, int(np.mean([sizes[w] for w in known])))
                for w in missing:
                    sizes[w] = fallback_size
                print(f"[{self.orchestrator_name}] [WARN] Dimensione shard non disponibile per {missing}: "
                      f"uso la media dei worker noti ({fallback_size}) come stima di fallback.")
            elif not known or sum(sizes[w] for w in known) <= 0:
                print(f"[{self.orchestrator_name}] [WARN] Nessuna dimensione di shard rilevata per alcun "
                      f"worker (probabile dataset sintetico): ricado sulla ripartizione EQUA storica.")
                sizes = {w: 1 for w in worker_names}

        total_size = sum(sizes.get(w, 0) for w in worker_names)
        if total_size <= 0:
            sizes = {w: 1 for w in worker_names}
            total_size = len(worker_names)

        raw_quotas = {w: total_step_trees * sizes[w] / total_size for w in worker_names}
        quotas = {w: int(np.floor(q)) for w, q in raw_quotas.items()}
        remainder = total_step_trees - sum(quotas.values())

        # Distribuiamo l'arrotondamento residuo ai worker con la parte
        # frazionaria più alta (metodo dei resti più grandi / Hamilton).
        by_fraction_desc = sorted(worker_names, key=lambda w: raw_quotas[w] - quotas[w], reverse=True)
        for w in by_fraction_desc[:remainder]:
            quotas[w] += 1

        label = "EQUA" if strategy == "equal" else "proporzionale alla dimensione dello shard"
        print(f"[{self.orchestrator_name}] Allocazione alberi {label}: " +
              ", ".join(f"{w}={quotas[w]} (size={sizes.get(w, '?')})" for w in worker_names))
        return quotas

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> int:
        """
        Invia la richiesta a ciascun worker attivo per il proprio shard locale.
        Se un worker fallisce, viene registrato il dropout e si prosegue con i rimanenti.
        Alla fine, se gli alberi totali superano il target (over-provisioning), applica lo scarto uniforme.
        """
        self.current_job_id = payload.get("job_id")

        # Azzerati a ogni step, come nel centralizzato: senza questo, uno step
        # che non costruisce alberi (o che fallisce prima del dispatch)
        # lascerebbe in piedi i valori dello step PRECEDENTE, e il report li
        # attribuirebbe a questo. Eseguendo 'all' gli scenari condividono la
        # stessa istanza di orchestratore, quindi il rischio e' concreto.
        self.last_dispatch_seconds = 0.0
        self.last_aggregation_seconds = 0.0

        checkpoint_trees_path = self._resolve_trees_checkpoint_path(self.current_job_id)
        if self.environment == "aws":
            self._ensure_aws_bootstrap(payload)
        else:
            self._ensure_local_bootstrap(payload)
            os.makedirs("./.local_storage", exist_ok=True)


        if start_alberi == 0:
            # Rimuove parti incrementali E monolitico: ripartendo da zero non
            # deve sopravvivere nulla di un tentativo precedente sullo stesso id.
            self._purge_trees_checkpoint(self.current_job_id)
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
                if self._trees_checkpoint_exists(self.current_job_id):
                    try:
                        # Il checkpoint e' ora INCREMENTALE: una parte per scrittura,
                        # contenente solo gli alberi nuovi. _load_trees_checkpoint le
                        # rilegge in ordine e le ricompone, con fallback automatico sul
                        # vecchio formato monolitico se non esiste alcuna parte.
                        all_trained_trees = self._load_trees_checkpoint(self.current_job_id)
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
            # FIX: 'class_weight' era già supportato lato worker
            # (exposed_train_local_federated_forest / build_single_tree in
            # federatedWorker.py: se presente negli hyperparameters, viene
            # passato al DecisionTreeClassifier), ma non veniva MAI impostato
            # qui -- restava sempre None, cioè nessun bilanciamento per
            # nessun albero. Rilevante soprattutto con partition_strategy
            # non-IID come 'by_day', dove il rapporto Benign/Attacco di un
            # singolo shard può essere molto diverso da quello globale (es.
            # worker con 13 o 7 soli campioni della classe minoritaria):
            # 'balanced' pesa ogni classe in proporzione inversa alla sua
            # frequenza LOCALE, così anche gli alberi di un worker con shard
            # fortemente sbilanciato non collassano semplicemente sulla
            # classe locale maggioritaria. Rispetta comunque un valore
            # esplicito nel manifesto (hp), se presente; il default
            # 'balanced' si applica solo ai classificatori, mai alla
            # regressione (class_weight non ha senso lì).
            class_weight = hp.get("class_weight", "balanced" if tree_type == "classifier" else None)

            # FIX: con dataset_type='synthetic', ogni worker generava un
            # numero FISSO di campioni (166666, il fallback hardcoded in
            # _load_synthetic_data quando 'n_samples' non è tra gli
            # hyperparameters ricevuti) indipendentemente da quanti worker
            # sono effettivamente attivi -- coincidenza numerica con
            # 1.000.000/6, non un calcolo dinamico: con un numero di worker
            # diverso da 6, la somma dei dati "visti" dal federato smetteva
            # di corrispondere al totale usato dal centralizzato, rendendo i
            # due esperimenti non più confrontabili sullo stesso volume dati.
            # Qui leggiamo il totale da config_synthetic.json (stesso valore
            # che il centralizzato usa per intero) e lo ripartiamo in quote
            # il più possibile uguali tra i worker REALMENTE attivi in questo
            # round: la somma torna sempre 'total_n_samples' esatto,
            # qualunque sia NUM_WORKERS. None (nessuna riga 'n_samples' nel
            # config, o config assente) -> nessun override esplicito, ogni
            # worker ricade sul proprio fallback fisso come accadeva prima.
            synthetic_n_samples_by_worker = {}
            if self._resolve_dataset_type(payload) == "synthetic":
                total_synthetic_n_samples = self._read_synthetic_total_n_samples()
                if total_synthetic_n_samples is not None:
                    synthetic_n_samples_by_worker = self._allocate_synthetic_samples(
                        worker_names, total_synthetic_n_samples
                    )
                    print(f"[{self.orchestrator_name}] Dataset sintetico: {total_synthetic_n_samples} campioni "
                          f"totali ripartiti tra {num_workers} worker attivi "
                          f"(~{total_synthetic_n_samples // num_workers} campioni/worker).")

            # Iperparametro dell'ESPERIMENTO (non del modello): come sono stati
            # ripartiti i dati tra i worker in fase di provisioning. Letto qui
            # solo per tracciabilità nei log/nelle metriche — la ripartizione
            # vera e propria è già avvenuta offline (provision_*_shards.py); qui
            # ne teniamo semplicemente traccia per poter correlare i risultati
            # del job con la strategia/alpha usati per generare gli shard.
            # Campi PIATTI su TrainingRequest (non nidificati sotto
            # "hyperparameters"), popolati da main.py leggendo il manifesto:
            # sopravvivono al giro completo client -> SQS -> qui.
            partitioning_info = {
                "strategy": payload.get("partition_strategy", "iid"),
                "alpha": payload.get("partition_alpha"),
                "tree_allocation": payload.get("tree_allocation_strategy", "proportional"),
            }
            print(f"[{self.orchestrator_name}] Partizionamento federato dichiarato nel manifesto: "
                  f"strategy='{partitioning_info.get('strategy', 'iid')}'"
                  + (f", alpha={partitioning_info.get('alpha')}" if partitioning_info.get("strategy") == "dirichlet" else "")
                  + f" | tree_allocation='{partitioning_info.get('tree_allocation')}'.")

            # Allocazione del budget di alberi tra i worker, secondo la strategia
            # dichiarata (vedi _fetch_worker_shard_sizes / _allocate_tree_quotas).
            # Con "equal" saltiamo del tutto la probe RPC delle dimensioni: non
            # serve, e risparmia un giro di rete per ogni round di training.
            tree_allocation_strategy = partitioning_info["tree_allocation"]
            if tree_allocation_strategy == "equal":
                worker_shard_sizes = {}
            else:
                worker_shard_sizes = self._fetch_worker_shard_sizes(
                    worker_names, available_workers, job_id=self.current_job_id
                )
            tree_quotas = self._allocate_tree_quotas(
                total_step_trees, worker_names, worker_shard_sizes, strategy=tree_allocation_strategy
            )

            assigned_tasks = {}
            sub_start = start_alberi
            task_id_counter = start_alberi + 1
            # riceve UN SOLO chunk, di sua esclusiva proprietà. Se il worker muore mentre
            # lo sta processando, il chunk NON viene ripreso da nessun altro worker:
            # il thread dedicato smette di ritentare la RPC e resta in attesa che quello
            # stesso worker ricompaia nel ServiceRegistry, poi riprova lo stesso task.
            for w_name in worker_names:
                quota_chunk = tree_quotas.get(w_name, 0)
                if quota_chunk <= 0:
                    print(f"[{self.orchestrator_name}] [SKIP] '{w_name}' non riceve alberi in questo round "
                          f"(quota proporzionale = 0).")
                    continue
                sub_end = sub_start + quota_chunk
                task_seed = seed + sub_start
                assigned_tasks[w_name] = (task_id_counter, sub_start, sub_end, task_seed)
                task_id_counter += 1
                sub_start = sub_end

            feature_selezionate = (None if self.environment == "aws" else self.select_from_config(self._resolve_dataset_type(payload)))
            results_lock = threading.Lock()
            checkpoint_time_accum = [0.0]

            # Lock DEDICATO alla persistenza del checkpoint, separato da
            # results_lock. Prima l'upload su S3 avveniva dentro results_lock,
            # cioe' dentro la stessa sezione critica che serve ad accodare gli
            # alberi ricevuti: ogni worker che finiva restava fermo ad aspettare
            # la fine dell'upload di un altro, non per calcolare ma solo per
            # poter registrare il proprio risultato. Era un punto di
            # serializzazione che cresceva col numero di worker, e falsava
            # proprio la misura di strong scaling.
            checkpoint_lock = threading.Lock()
            # Contatore monotono dell'ultimo snapshot effettivamente persistito:
            #  1) impedisce che uno snapshot piu' VECCHIO sovrascriva uno piu'
            #     recente, ora che la scrittura e' fuori da results_lock e due
            #     thread possono arrivarci in ordine diverso da quello in cui
            #     hanno preso lo snapshot (un checkpoint che regredisce
            #     sposterebbe INDIETRO il punto di ripartenza dopo un guasto);
            #  2) salta le scritture gia' superate: se risulta persistito uno
            #     stato con piu' alberi, riscrivere e' inutile e il checkpoint
            #     resta comunque piu' avanti.
            # "parts" riparte dal numero di parti gia'su storage: scrivere di nuovo
            # dalla 0 sovrascriverebbe un delta valido con un altro delta.
            checkpoint_lock_state = {"count": start_alberi,
                                     "parts": self._count_trees_checkpoint_parts(self.current_job_id)}
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
                                'sync_request_timeout': RPC_SYNC_TIMEOUT_SECONDS,
                                'keepalive': True
                            }
                        )
                        with self.connessioni_lock:
                            self.connessioni_attive.append(worker_conn)
                        print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="PROCESSING")
                        self.chunk_sent_event.set()
                        effective_seed = chunk_seed + (idx * 1000)
                        ack_raw = worker_conn.root.exposed_train_local_federated_forest(
                            job_id=self.current_job_id,
                            dataset_type=self._resolve_dataset_type(payload),
                            n_estimators_local=quota_chunk,
                            worker_index=idx,
                            hyperparameters={
                                **hp,
                                "random_state": effective_seed,
                                "dataset_random_state": seed,
                                "feature_selezionate": feature_selezionate,
                                "class_weight": class_weight,
                                # Presente solo per dataset_type='synthetic' e solo
                                # se config_synthetic.json aveva 'n_samples' (vedi
                                # sopra): quota di QUESTO worker sul totale
                                # sintetico condiviso, invece del fallback fisso
                                # indipendente dal numero di worker.
                                **({"n_samples": synthetic_n_samples_by_worker[w_name]}
                                   if w_name in synthetic_n_samples_by_worker else {}),

                            },
                        )

                        # Il worker NON restituisce più gli alberi per intero via
                        # RPC: li ha già persistiti nello storage condiviso prima
                        # di rispondere (vedi federatedWorker.py,
                        # exposed_train_local_federated_forest), e qui ci
                        # limitiamo a un piccolo ack + rilettura diretta dallo
                        # storage. Stesso fix già applicato al path centralizzato
                        # per evitare l'hang osservato quando RPyC deve
                        # trasportare un payload sincrono molto grande come
                        # valore di ritorno (Scenario 2 - Scalabilità).
                        ack = obtain(ack_raw)
                        if not isinstance(ack, dict) or not ack.get("ack"):
                            raise RuntimeError(
                                f"Risposta inattesa dal worker {w_name} per il task {task_id}: {ack!r}"
                            )

                        # 'source_info' sintetico: il federato non usa un path di
                        # dataset per il training locale (ogni worker ha già il
                        # proprio shard in cache), quindi costruiamo la stessa
                        # stringa 'shared_train_<job_id>.csv' usata lato worker
                        # per derivare la medesima chiave di storage.
                        #
                        # 'load_task_trees_from_shared_storage' ritorna gli alberi
                        # già deserializzati, evitando il giro superfluo
                        # oggetti->bytes->oggetti che load_task_from_shared_storage
                        # avrebbe richiesto qui (stesso fix applicato al path
                        # centralizzato dopo l'OOM osservato sull'Orchestratore).
                        #
                        # 'tree_reconstruction_lock' (definito in BaseOrchestrator,
                        # condiviso con il path centralizzato) serializza QUESTA fase
                        # tra i thread worker, per evitare che più task vengano
                        # ricomposti in RAM nello stesso istante -- stesso fix
                        # applicato a centralized.py dopo l'OOM osservato anche con
                        # 16GB di memoria sull'Orchestratore.
                        synthetic_source_info = f"shared_train_{self.current_job_id}.csv"
                        with self.tree_reconstruction_lock:
                            result_trees = load_task_trees_from_shared_storage(
                                synthetic_source_info, effective_seed, quota_chunk,
                                self.environment, self.orchestrator_name
                            )
                        if result_trees is None:
                            raise RuntimeError(
                                f"Worker {w_name}: task {task_id} confermato (ack) ma il blob "
                                f"non è stato trovato nello storage condiviso."
                            )
                        # SEZIONE CRITICA MINIMA: solo l'aggiornamento della lista
                        # condivisa e uno snapshot immutabile. Upload su S3 e
                        # scrittura su DynamoDB sono spostati fuori (vedi sotto).
                        with results_lock:
                            all_trained_trees.extend(result_trees)
                            current_total = len(all_trained_trees)
                            # Copia: la serializzazione fuori dal lock non deve
                            # poter vedere la lista mutare sotto i piedi.
                            snapshot = list(all_trained_trees)

                        # --- fuori da results_lock ---
                        with checkpoint_lock:
                            if current_total > checkpoint_lock_state["count"]:
                                try:
                                    t_chk_start = time.perf_counter()
                                    # Scrive SOLO gli alberi nuovi (alla parte 0 l'intero
                                    # snapshot, per migrare dal formato monolitico).
                                    self._persist_trees_delta(
                                        self.current_job_id, snapshot,
                                        checkpoint_lock_state["count"], checkpoint_lock_state["parts"])
                                    checkpoint_time_accum[0] += time.perf_counter() - t_chk_start
                                    checkpoint_lock_state["count"] = current_total
                                    checkpoint_lock_state["parts"] += 1
                                    # Cache di istanza allineata SOLO dopo il salvataggio
                                    # riuscito, cosi' non e' mai "piu' avanti" della fonte
                                    # di verita' persistita.
                                    self._trees_cache[self.current_job_id] = snapshot
                                    print(f"   [RPC <- {w_name}] [CHECKPOINT FS OK] Task {task_id} archiviato. Progressivo in RAM/Storage: {current_total} alberi.")
                                except Exception as e_fs:
                                    # Il contatore NON avanza: un writer successivo deve
                                    # poter riprovare a persistere lo stato.
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
                            else:
                                print(f"   [RPC <- {w_name}] [CHECKPOINT SKIP] Task {task_id}: gia' persistito uno "
                                      f"stato piu' avanzato ({checkpoint_lock_state['count']} alberi >= {current_total}).")

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
            dispatch_start = time.perf_counter()
            threads = []
            for i, worker_name in enumerate(worker_names, start=1):
                stable_idx = self._infer_worker_index(worker_name, i)
                t = threading.Thread(target=contact_worker, args=(worker_name, stable_idx))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            self.last_dispatch_seconds = time.perf_counter() - dispatch_start
            print(f"[DEBUG] Tempo totale speso in I/O di checkpoint: {checkpoint_time_accum[0]:.2f}s")

            if not all_trained_trees:
                raise RuntimeError("Tutti i nodi interessati sono falliti. Nessun albero raccolto per questo Job.")

            if len(all_trained_trees) > target_alberi:
                print(f"[{self.orchestrator_name}] [SCARTO UNIFORME] Trovati {len(all_trained_trees)} alberi. Riduzione casuale a quota {target_alberi}.")
                collected_trees = random.sample(all_trained_trees, target_alberi)
            else:
                print(f"[{self.orchestrator_name}] Raccolti in totale {len(all_trained_trees)} alberi dai worker superstiti.")
                collected_trees = all_trained_trees

            aggregation_start = time.perf_counter()
            # NB: 'collected_trees' viene svuotata (elementi -> None) DENTRO
            # questa chiamata, man mano che ogni albero viene persistito su
            # disco, per liberare RAM il prima possibile invece di aspettare
            # la fine dell'intero ciclo (vedi il commento nel corpo della
            # funzione). Per questo 'detected_classes' viene ora restituita
            # da qui invece di essere ricalcolata dopo da 'collected_trees'
            # (che a quel punto sarebbe vuota, e la calibrazione sotto
            # sarebbe silenziosamente saltata per ogni classificatore).
            final_count, detected_classes = self._reconstruct_and_save_global_model(collected_trees, tree_type)
            self.last_aggregation_seconds = time.perf_counter() - aggregation_start

            if tree_type == "classifier":
                try:
                    if detected_classes is not None and len(detected_classes) > 0:
                        global_classes = detected_classes
                        # Ri-query fresca (invece di riusare 'available_workers' catturato
                        # prima del dispatch di training): un worker può aver cambiato
                        # host/porta nel frattempo, e la calibrazione non è così urgente
                        # da giustificare il rischio di contattare un endpoint stantio.
                        calibration_workers = ServiceRegistry.get_available_workers(self.environment)
                        self._calibrate_federated_threshold(self.current_job_id, global_classes, calibration_workers, final_count)
                except Exception as e_thr:
                    print(f"[{self.orchestrator_name}] [WARN] Calibrazione soglia federata fallita "
                          f"(non bloccante, l'inferenza ricadrà sulla soglia della baseline): {e_thr}")

            # FIX MEMORIA: NON passare più 'alberi_reali=collected_trees' qui.
            # Il commento originale in _save_checkpoint diceva esplicitamente che
            # quel ramo era pensato per non essere MAI esercitato in condizioni
            # normali ("BaseOrchestrator chiama _save_checkpoint senza
            # 'alberi_reali'") -- ma passandolo qui veniva invece eseguito ad
            # OGNI round completato con successo, ri-serializzando l'INTERA
            # foresta un'altra volta (dopo che era già stata scritta un albero
            # alla volta in model_dir da _reconstruct_and_save_global_model, e
            # dopo che era già stata salvata incrementalmente durante il round
            # da _persist_trees_delta ad ogni task completato). Tre copie
            # ridondanti dello stesso lavoro, proprio nel momento in cui
            # 'collected_trees' occupa già il picco di RAM del round: è una
            # delle cause dell'OOM osservato sull'orchestratore. Il checkpoint
            # leggero (stato/contatori, senza gli alberi) resta comunque
            # salvato da super()._save_checkpoint(...) dentro _save_checkpoint.
            self._save_checkpoint(self.current_job_id, final_count, payload.get("retries", 0), seed)

            # Libera subito la cache in-memoria di questo job: serviva solo per
            # evitare un reload da storage se lo STESSO processo gestisce il
            # round successivo (vedi 'STATE-SYNC' più sopra), ma il round è
            # appena terminato con successo e il modello è già durevolmente su
            # model_dir -- non c'è motivo di tenere ~100 alberi vivi in RAM
            # per il resto della vita del processo (es. mentre parte
            # l'inferenza subito dopo, come nei test).
            self._trees_cache.pop(self.current_job_id, None)

            return final_count

    def _execute_inference_step(self, payload: dict) -> dict:
        print(f"\n[{self.orchestrator_name}] == AVVIO VALIDAZIONE FEDERATA DISTRIBUITA ==")
        job_id = payload.get("job_id")
        hyperparameters = payload.get("hyperparameters", {})
        tree_type = hyperparameters.get("tree_type", "classifier")
        # Stesso iperparametro dell'esperimento letto in fase di training: qui
        # serve solo per taggare le metriche salvate, non cambia il comportamento
        # dell'inferenza (che è comunque agnostica alla strategia di sharding
        # usata a monte, valuta semplicemente il modello globale già assemblato).
        # Campi piatti su InferenceRequest, popolati da main.py a partire dallo
        # storico locale del job di training corrispondente.
        partitioning_info = {
            "strategy": payload.get("partition_strategy", "iid"),
            "alpha": payload.get("partition_alpha"),
            "tree_allocation": payload.get("tree_allocation_strategy", "proportional"),
        }

        # Marcatore di "inferenza avviata": l'inferenza federata calcola tutto in
        # RAM e via RPC, senza salvare il checkpoint chunk-per-chunk che invece
        # produce quella centralizzata. Non lascerebbe quindi alcun segnale
        # osservabile dall'esterno mentre è in corso — il che rende impossibile,
        # per un monitor esterno (o per il test di failover), sapere che l'inferenza
        # è davvero partita. Scriviamo un flag leggero su disco per colmare questo
        # divario: viene creato appena l'inferenza inizia e rimosso quando termina.
        self._mark_inference_started(job_id)

        inference_start_time = time.perf_counter()

        # FIX allineamento path: qui si ricostruiva il path a mano con
        # os.path.join(self.models_dir, ...), un attributo non impostato da
        # questa classe (probabile residuo di un'implementazione precedente)
        # e comunque con una convenzione di naming diversa da quella usata in
        # fase di training (manca il prefisso 'model_' e, su AWS, il
        # sottopercorso 'saved_models/federated/'). Il training salva sempre
        # tramite _resolve_model_dir/_resolve_model_path (vedi più sotto in
        # questo file), quindi l'inferenza deve leggere dagli stessi identici
        # resolver, non da un path costruito in modo indipendente.
        trees_dir = self._resolve_model_dir(job_id)
        model_pkl = self._resolve_model_path(job_id)
        meta_path = self._resolve_model_meta_path(job_id)

        # NB: il controllo di esistenza deve passare SEMPRE da checkpoint_dao,
        # mai da os.path.exists/os.path.isdir/os.listdir direttamente: questi
        # ultimi funzionano solo su path locali, mentre su 'aws' trees_dir/
        # model_pkl sono URI 's3://...' (os.path.exists su un URI del genere
        # ritorna silenziosamente False, senza errore, quindi il blocco
        # solleverebbe FileNotFoundError anche quando il modello esiste
        # davvero su S3). CheckpointDAO inoltre non espone alcun metodo di
        # listing su S3, quindi non possiamo comunque "contare i file nella
        # cartella" come si faceva prima: usiamo la presenza dei metadati
        # (scritti dal training in modalità streaming, vedi model_dir più
        # sotto in questo file) come segnale di esistenza del modello a
        # alberi separati, che è equivalente e non richiede listing.
        if self.checkpoint_dao.exists(meta_path):
            is_streaming = True
        elif self.checkpoint_dao.exists(model_pkl):
            is_streaming = False
        else:
            raise FileNotFoundError(
                f"Nessun modello trovato per il job '{job_id}'. "
                f"Cercati: metadati '{meta_path}' (modalità a-alberi-separati "
                f"in '{trees_dir}') o file singolo '{model_pkl}'"
            )
        # Leggiamo solo i metadati leggeri (conteggio alberi, classi) invece di
        # deserializzare l'intero modello: qui serve solo per loggare/taggare
        # le metriche e per 'global_classes' nel payload ai worker — sono loro
        # (non l'orchestratore) a scaricare e deserializzare gli alberi veri e
        # propri per predire (vedi 'model_path' passato più sotto). Fallback al
        # caricamento completo per compatibilità con modelli salvati PRIMA di
        # questo fix (nessun file '.meta' ancora presente per quel job).
        global_classes = None
        if self.checkpoint_dao.exists(meta_path):
            meta = self.checkpoint_dao.load(meta_path)
            total_trees = meta["num_trees"]
            global_classes = meta.get("classes")
            print(f"[{self.orchestrator_name}] Metadati letti da {meta_path}. "
                  f"Numero totale di alberi: {total_trees}")
        else:
            print(f"[{self.orchestrator_name}] [WARN] Metadati leggeri non trovati per questo job "
                  f"(modello salvato prima di questo fix?): fallback al caricamento completo di "
                  f"{self._resolve_model_dir(job_id)}...")
            fallback_model = self.checkpoint_dao.load(self._resolve_model_path(job_id))
            total_trees = len(fallback_model.estimators_)
            if hasattr(fallback_model, "classes_"):
                global_classes = fallback_model.classes_.tolist()
            print(f"[{self.orchestrator_name}] Foresta caricata (fallback). Numero totale di alberi: {total_trees}")

        available_workers = ServiceRegistry.get_available_workers(self.environment)
        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        if num_workers == 0:
            raise RuntimeError("Nessun worker disponibile per l'inferenza federata.")
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        # Non serializziamo più l'intera foresta per rimandarla via RPC: il
        # modello è già su storage condiviso a 'model_path' (l'abbiamo appena
        # caricato da lì), quindi passiamo solo quel riferimento a ciascun
        # worker, che lo scarica e deserializza da sé. Prima veniva rifatto
        # pickle.dumps(all_trees) una volta qui E il blob risultante (fino a
        # 1+ GB) veniva ritrasmesso per intero via RPC UNA VOLTA PER OGNI
        # WORKER — peggio ancora del path centralizzato, dove almeno la
        # foresta viene divisa in chunk tra i worker invece di essere ripetuta.
        feature_selezionate = (
            None if self.environment == "aws"
            else self.select_from_config(self._resolve_dataset_type(payload))
        )
        # Soglia di decisione: proviamo PRIMA quella calibrata specificamente sul
        # modello federato (vedi _calibrate_federated_threshold, eseguita a fine
        # training se tree_type='classifier') -- calcolata sul validation set
        # federato con la STESSA funzione di scoring (voto pesato per foglia)
        # usata qui in inferenza, quindi più rappresentativa di quella della
        # baseline centralizzata. Se assente (calibrazione fallita, job
        # addestrato prima di questa modifica, o modello non-classificatore),
        # ricadiamo sulla soglia della baseline (comportamento storico, invariato).
        decision_threshold = None
        threshold_path = self._resolve_threshold_path(job_id)
        if self.checkpoint_dao.exists(threshold_path):
            try:
                thr_data = self.checkpoint_dao.load(threshold_path)
                decision_threshold = thr_data.get("decision_threshold")
                print(f"[{self.orchestrator_name}] Soglia FEDERATA calibrata trovata per questo job: "
                      f"{decision_threshold:.4f} (calcolata su {thr_data.get('n_val_samples')} "
                      f"campioni di validation, F1={thr_data.get('f1_score')}).")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Soglia federata presente ma illeggibile "
                      f"({e}): ricado sulla soglia della baseline.")

        if decision_threshold is None:
            # Vedi VALIDATION_SIZE_FOR_THRESHOLD/decision_threshold in run_baseline.py:
            # stesso pattern di feature_selezionate sopra, incluso il guard su AWS
            # (il file locale config_<dataset_type>.json non esiste sul container
            # dell'orchestratore in quell'ambiente). None -> ogni worker ricade sul
            # comportamento di default (argmax/soglia implicita 0.50), vedi
            # FederatedWorker.exposed_predict_subset_forest.
            decision_threshold = (
                None if self.environment == "aws"
                else self.read_decision_threshold_from_config(self._resolve_dataset_type(payload))
            )
            if decision_threshold is not None:
                print(f"[{self.orchestrator_name}] [FALLBACK] Nessuna soglia federata calibrata "
                      f"disponibile per questo job: uso quella della baseline ({decision_threshold}).")

        # SOGLIE PER-WORKER (vedi _calibrate_federated_threshold): con
        # partizionamento non-IID la soglia unica sopra può essere lontana
        # dall'ottimo per un worker la cui distribuzione locale delle
        # probabilità è spostata rispetto al pool. Se disponibili, ogni
        # worker riceve la PROPRIA soglia invece di quella broadcast a tutti
        # -- fallback su 'decision_threshold' (sopra) per qualunque worker
        # senza una soglia individuale salvata (validation locale
        # insufficiente, o job addestrato prima di questa modifica).
        per_worker_thresholds = {}
        per_worker_threshold_path = self._resolve_per_worker_threshold_path(job_id)
        if self.checkpoint_dao.exists(per_worker_threshold_path):
            try:
                per_worker_thresholds = self.checkpoint_dao.load(per_worker_threshold_path)
                print(f"[{self.orchestrator_name}] Soglie per-worker trovate per {len(per_worker_thresholds)} "
                      f"worker: {sorted(per_worker_thresholds.keys())}. Gli altri useranno la soglia globale "
                      f"({decision_threshold}).")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Soglie per-worker presenti ma illeggibili "
                      f"({e}): tutti i worker useranno la soglia globale.")
                per_worker_thresholds = {}

        # Accumulo per-worker (non più liste piatte): la chiave è il worker_index
        # STABILE (lega worker<->shard, vedi _infer_worker_index), così la ripresa
        # dopo un failover sa esattamente quali worker sono già stati validati e
        # può saltarli, richiedendo solo quelli mancanti. Ogni voce contiene il
        # risultato completo di un worker: {y_pred, y_true, y_probs, n_samples}.
        # Struttura persistita tramite CheckpointDAO in inference_chunks_{job_id}.
        results_by_worker = {}
        failed_workers = set()
        self.chunk_sent_event.clear()
        results_lock = threading.Lock()
        INF_RETRY_WAIT_SECONDS = 10

        # Ripresa da checkpoint: se un leader precedente (poi caduto) aveva già
        # raccolto i risultati di alcuni worker, li ricarichiamo e non li
        # richiediamo di nuovo. In inferenza federata "un worker" è l'unità di
        # lavoro (predice sul proprio shard), quindi il checkpoint è chiavato per
        # worker_index, non per chunk di alberi come nel centralizzato.
        inference_cp_path = self._get_inference_checkpoint_path(job_id)
        try:
            if self.checkpoint_dao.exists(inference_cp_path):
                restored = self.checkpoint_dao.load(inference_cp_path)
                if isinstance(restored, dict):
                    results_by_worker.update(restored)
                    print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Ripristinati "
                          f"{len(results_by_worker)} worker già validati dal checkpoint: "
                          f"{sorted(results_by_worker.keys())}.")
        except Exception as e:
            print(f"[{self.orchestrator_name}] [WARN] Checkpoint di inferenza non caricabile "
                  f"({e}): riparto senza ripresa.")

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
                        config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": RPC_INFERENCE_SYNC_TIMEOUT_SECONDS}
                    )
                    with self.connessioni_lock:
                        self.connessioni_attive.append(conn)
                    self.chunk_sent_event.set()

                    worker_hyperparameters = {
                        **hyperparameters,
                        "dataset_type": self._resolve_dataset_type(payload),
                        "feature_selezionate": feature_selezionate,
                        "tree_type": tree_type,
                        # Soglia individuale se calibrata per QUESTO worker
                        # (vedi per_worker_thresholds sopra), altrimenti quella
                        # globale broadcast a tutti come prima di questa modifica.
                        "decision_threshold": per_worker_thresholds.get(idx, {}).get(
                            "decision_threshold", decision_threshold
                        ),
                        "num_trees": total_trees,
                    }
                    if tree_type == "classifier" and global_classes is not None:
                        worker_hyperparameters["global_classes"] = global_classes
                    # ----------------------------------------------------------------------------
                    #print(f"[{self.orchestrator_name}-InfThread] Invio riferimento al modello globale "
                          #f"({total_trees} alberi, {model_path}) a {w_name}...")
                    raw_response = conn.root.exposed_predict_subset_forest(payload=pickle.dumps({
                        #"model_path": model_path,
                        "model_dir"  : trees_dir,
                        "job_id": job_id,
                        "worker_index": idx,
                        "hyperparameters": worker_hyperparameters
                    }))
                    worker_data = pickle.loads(obtain(raw_response))

                    with results_lock:
                        # Registriamo il risultato COMPLETO del worker sotto il suo
                        # indice stabile. Salvare per-worker (invece di estendere
                        # liste piatte) è ciò che rende la ripresa possibile: un
                        # eventuale standby subentrato ritrova esattamente questi
                        # risultati e non re-interroga i worker già validati.
                        worker_probs = worker_data.get("y_probs") if tree_type == "classifier" else None
                        if tree_type == "classifier" and worker_probs is None:
                            print(f"[{self.orchestrator_name}] [WARN] Worker '{w_name}' non ha restituito "
                                  f"'y_probs': l'AUC finale sarà None (worker non aggiornato).")
                        results_by_worker[idx] = {
                            "y_pred": list(worker_data["y_pred"]),
                            "y_true": list(worker_data["y_true"]),
                            "y_probs": list(worker_probs) if worker_probs is not None else None,
                            "n_samples": worker_data["n_samples"],
                        }
                        # Persistiamo il checkpoint aggiornato. La LocalCheckpointDAO
                        # scrive in modo atomico (tmp + os.replace), quindi un crash
                        # a metà non corrompe il file già valido.
                        try:
                            self.checkpoint_dao.save(inference_cp_path, dict(results_by_worker))
                        except Exception as cp_err:
                            print(f"[{self.orchestrator_name}] [WARN] Salvataggio checkpoint inferenza "
                                  f"fallito per worker {idx}: {cp_err}")
                        print(f"[{self.orchestrator_name}] Validazione completata su '{w_name}' "
                              f"({worker_data['n_samples']} record). Checkpoint: {len(results_by_worker)} worker.")
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
            stable_idx = self._infer_worker_index(name, idx)
            # SHORT-CIRCUIT ripresa: se questo worker è già nel checkpoint (validato
            # da un leader precedente prima del crash), non lo re-interroghiamo.
            if stable_idx in results_by_worker:
                print(f"[{self.orchestrator_name}] [SHORT-CIRCUIT INF] Worker '{name}' "
                      f"(index {stable_idx}) già validato dal checkpoint. Skip.")
                continue
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

        # Assemblaggio finale: ricomponiamo le liste globali dai risultati
        # per-worker (sia quelli ripresi dal checkpoint sia quelli appena
        # raccolti). Ordiniamo per worker_index così l'output è deterministico
        # e y_probs resta allineato a y_pred/y_true campione-per-campione.
        y_pred_global = []
        y_true_global = []
        y_probs_global = []
        total_samples = 0
        all_probs_present = True
        for w_idx in sorted(results_by_worker.keys()):
            r = results_by_worker[w_idx]
            y_pred_global.extend(r["y_pred"])
            y_true_global.extend(r["y_true"])
            total_samples += r["n_samples"]
            if r.get("y_probs") is not None:
                y_probs_global.extend(r["y_probs"])
            else:
                all_probs_present = False
        total_samples_ref = [total_samples]

        if not y_pred_global:
            print(f"[{self.orchestrator_name}] [ERRORE] Nessun worker ha risposto alla validazione federata.")
            self._clear_inference_started(job_id)
            return {}

        if failed_workers:
            print(f"[{self.orchestrator_name}] [WARN] {len(failed_workers)} worker non hanno risposto: {failed_workers}. Metriche calcolate sui rimanenti.")

        total_inference_time = time.perf_counter() - inference_start_time

        y_true_dtype = np.float64 if tree_type == "regressor" else np.int64

        # y_probs è utilizzabile per l'AUC solo se OGNI worker incluso lo ha
        # fornito: in tal caso è allineato campione-per-campione con
        # y_pred/y_true (stesso ordine per-worker). Se anche un solo worker non
        # l'ha restituito, l'array sarebbe disallineato: meglio non calcolare
        # l'AUC che calcolarla male.
        y_probs_array = None
        if tree_type == "classifier" and all_probs_present and len(y_probs_global) == len(y_pred_global):
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

        metrics_per_worker = {}
        for w_idx, r in results_by_worker.items():
            y_true_arr = np.array(r["y_true"], dtype=y_true_dtype)
            classes, counts = np.unique(y_true_arr, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))

            y_probs_w = np.array(r["y_probs"], dtype=np.float64) if r.get("y_probs") is not None else None
            try:
                m = self.calculate_metrics(
                    final_predictions=np.array(r["y_pred"], dtype=np.float64),
                    y_test=y_true_arr,
                    tree_type=tree_type,
                    y_probs=y_probs_w,
                )
            except ValueError as e:
                print(f"[{self.orchestrator_name}] [WARN] calculate_metrics fallito per worker {w_idx} "
                    f"(class_counts={class_counts}): {e}. Escluso dal report metriche.")
                continue  # niente entry per questo worker: meglio assente che un crash dell'intera inferenza

            m["n_samples"] = r["n_samples"]
            m["class_counts"] = class_counts
            metrics_per_worker[w_idx] = m
        MIN_SAMPLES_PER_CLASS = 30  # da tarare
        def _reliable_for_macro(m):
            counts = m.get("class_counts", {})
            return bool(counts) and min(counts.values()) >= MIN_SAMPLES_PER_CLASS

        reliable = {k: v for k, v in metrics_per_worker.items() if _reliable_for_macro(v)}
        excluded = set(metrics_per_worker) - set(reliable)
        if excluded:
            print(f"[{self.orchestrator_name}] [WARN] Worker esclusi dalla macro-average "
                f"(< {MIN_SAMPLES_PER_CLASS} campioni in almeno una classe): {sorted(excluded)}")

        numeric_keys = [k for k, v in next(iter(reliable.values()), {}).items() if isinstance(v, (int, float))] if reliable else []
        metrics_macro = {
            k: float(np.mean([m[k] for m in reliable.values() if m.get(k) is not None]))
            for k in numeric_keys
        }
        self._save_metrics(job_id, "inference", {
            "job_id": job_id, "mode": "federated", "phase": "inference",
            "tree_type": tree_type, "testing_set_size": total_samples_ref[0],
            "federated_partitioning": partitioning_info,
            "timings": {"total_inference_time": total_inference_time, "rpc_inference_time": rpc_inference_time},
            "metrics": metrics,                        # invariato
            "metrics_per_worker": metrics_per_worker,   # NUOVO
            "metrics_macro": metrics_macro,             # NUOVO
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

        # Job concluso: rimuoviamo sia il marcatore di "inferenza avviata" sia il
        # checkpoint di inferenza per-worker. Lasciarli confonderebbe un eventuale
        # rilancio dello stesso job_id (ripresa da uno stato ormai completo).
        self._clear_inference_started(job_id)
        try:
            self.checkpoint_dao.delete(inference_cp_path)
        except Exception:
            pass
        return {
            "status": "SUCCESS" if not failed_workers else "PARTIAL",
            "testing_set_size": total_samples_ref[0],
            "failed_workers": list(failed_workers),
            "total_inference_time": total_inference_time,
            "rpc_inference_time": rpc_inference_time,
            "metrics": metrics
        }

    def _reconstruct_and_save_global_model(self, all_trained_trees: list, tree_type: str) -> tuple:
        """Ritorna (numero_alberi_salvati, classi_rilevate).

        NB: per liberare RAM il prima possibile, questa funzione SVUOTA
        (imposta a None) gli elementi di 'all_trained_trees' non appena li ha
        persistiti su disco -- vedi commento più sotto. Il chiamante NON deve
        quindi rileggere gli alberi da questa stessa lista dopo la chiamata
        (es. per le classi del classificatore): per questo la funzione
        restituisce già 'detected_classes', calcolate PRIMA dello svuotamento.
        """
        if not all_trained_trees:
            print(f"[{self.orchestrator_name}] Nessun albero collezionato.")
            return 0, np.array([0, 1])
        try:
            trees_with_classes = [t for t in all_trained_trees if hasattr(t, "classes_")]
            if trees_with_classes:
                detected_classes = np.unique(np.concatenate([np.asarray(t.classes_) for t in trees_with_classes]))
            else:
                detected_classes = np.array([0, 1])

            # Cartella dedicata per QUESTO job: un file pickle per albero invece
            # di un unico blob monolitico. Ogni file è piccolo (pochi MB), quindi
            # in inferenza si può caricarne un sottoinsieme alla volta senza mai
            # materializzare tutti i 100 alberi insieme in RAM.
            model_dir = self._resolve_model_dir(self.current_job_id)
            total_trees = len(all_trained_trees)
            # FIX MEMORIA: prima questo ciclo scriveva ogni albero su disco ma
            # lasciava TUTTI i riferimenti vivi nella lista 'all_trained_trees'
            # fino alla fine -- quindi per tutta la durata del ciclo la RAM
            # doveva comunque contenere l'intera foresta (100 alberi in
            # questo test), proprio nel momento di picco. Sostituendo ogni
            # elemento con None non appena è stato persistito, il refcount
            # dell'albero scende a zero e il garbage collector di CPython può
            # liberarlo subito, invece che solo all'uscita dalla funzione.
            # gc.collect() periodico (ogni 10 alberi, stesso batch_size usato
            # per lo streaming in inferenza in federatedWorker.py) forza il
            # rilascio effettivo della memoria invece di aspettare che il GC
            # generazionale ci arrivi da solo.
            for i in range(total_trees):
                tree = all_trained_trees[i]
                tree_path = os.path.join(model_dir, f"tree_{i:04d}.pkl")
                self.checkpoint_dao.save(tree_path, tree)
                all_trained_trees[i] = None
                if (i + 1) % 10 == 0:
                    gc.collect()
            gc.collect()

            meta = {
                "num_trees": total_trees,
                "tree_type": tree_type,
                "classes": detected_classes.astype(np.int64).tolist(),
                "model_dir": model_dir,   # esplicito, invece che dedurlo da job_id lato worker
            }
            self.checkpoint_dao.save(self._resolve_model_meta_path(self.current_job_id), meta)
            print(f"[{self.orchestrator_name}] Modello Globale salvato come {total_trees} "
                f"file separati in '{model_dir}'.")
            return total_trees, detected_classes
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE AGGREGAZIONE] {e}")
            traceback.print_exc()
            return len([t for t in all_trained_trees if t is not None]), np.array([0, 1])

    def _resolve_model_dir(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/federated/model_{job_id}_trees"
        return os.path.join("./saved_models", f"model_{job_id}_trees")

    def _resolve_threshold_path(self, job_id: str) -> str:
        """Path della soglia di decisione calibrata specificamente sul modello
        federato (vedi _calibrate_federated_threshold), in una sotto-cartella
        dedicata come model_path/model_meta_path. Assente per i job su cui la
        calibrazione non è stata eseguita o è fallita: in quel caso
        _execute_inference_step ricade sulla soglia della baseline centralizzata
        (comportamento storico, invariato)."""
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/federated/threshold_{job_id}.pkl"
        return os.path.join("./saved_models", f"threshold_{job_id}.pkl")

    def _resolve_per_worker_threshold_path(self, job_id: str) -> str:
        """Path del dizionario {stable_worker_idx: {decision_threshold, ...}}
        con le soglie calibrate individualmente per ciascun worker (vedi
        _calibrate_federated_threshold). Assente per i job su cui non è stato
        possibile calibrare NESSUN worker individualmente (validation set
        locale insufficiente ovunque): in quel caso _execute_inference_step
        usa la soglia globale per tutti, come già avveniva prima di questa
        modifica."""
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/federated/per_worker_threshold_{job_id}.pkl"
        return os.path.join("./saved_models", f"per_worker_threshold_{job_id}.pkl")

    def _calibrate_federated_threshold(self, job_id: str, global_classes, available_workers: dict, num_trees: int = None) -> float:
        """
        Calibra la soglia di decisione sul validation set FEDERATO (pool dei
        fold locali di ciascun worker -- self._cached_X_val/_y_val lato
        worker, mai visti in training né nel test finale), invece di riusare
        la soglia calibrata sul modello baseline centralizzato: non è detto
        sia più valida qui, specie dopo aver introdotto il voto pesato per
        foglia (vedi _weighted_forest_predict_proba in federatedWorker.py),
        che cambia la calibrazione delle probabilità restituite dal modello.

        Chiamata subito dopo l'aggregazione di un round di training (vedi
        _execute_training_step), quando il modello globale è già stato
        salvato su storage condiviso e i worker sono ancora raggiungibili.
        Non bloccante: in caso di fallimento (validation set insufficiente,
        RPC fallite, più di 2 classi globali) ritorna None e
        _execute_inference_step ricadrà sulla soglia della baseline, come
        avveniva prima di questa modifica.
        """
        if len(global_classes) != 2:
            print(f"[{self.orchestrator_name}] [CALIBRAZIONE SOGLIA] {len(global_classes)} classi globali "
                  f"rilevate: la calibrazione della soglia è supportata solo per il caso binario. "
                  f"Nessuna soglia federata calcolata.")
            return None

        # FIX: prima si passava 'model_path' (il vecchio blob monolitico),
        # ma da quando il training salva un file per albero in 'model_dir'
        # (vedi _reconstruct_and_save_global_model), 'model_path' non viene
        # più scritto su disco -- il worker lo avrebbe cercato invano. Il
        # caricamento va fatto in streaming da model_dir/num_trees, come già
        # avviene per l'inferenza vera e propria (exposed_predict_subset_forest).
        model_dir = self._resolve_model_dir(job_id)
        if num_trees is None:
            # Fallback: se il chiamante non lo passa esplicitamente (es. in
            # futuro venisse invocata da un altro punto del codice), lo
            # leggiamo dai metadati appena scritti.
            try:
                meta = self.checkpoint_dao.load(self._resolve_model_meta_path(job_id))
                num_trees = meta.get("num_trees")
            except Exception:
                num_trees = None
        if not num_trees:
            print(f"[{self.orchestrator_name}] [CALIBRAZIONE SOGLIA] Impossibile determinare 'num_trees' "
                  f"per il job '{job_id}'. Nessuna soglia federata calcolata.")
            return None

        all_y_true, all_y_probs = [], []
        per_worker_val = {}  # stable_idx -> (y_true list, y_probs list)
        results_lock = threading.Lock()

        def _probe(w_name, w_info, idx):
            conn = None
            try:
                conn = rpyc.connect(
                    w_info["host"], w_info["port"],
                    config={"allow_pickle": True, "sync_request_timeout": 120}
                )
                raw = conn.root.exposed_get_validation_predictions(pickle.dumps({
                    "model_dir": model_dir,
                    "num_trees": num_trees,
                    "global_classes": np.asarray(global_classes).tolist(),
                }))
                result = pickle.loads(obtain(raw))
                if result.get("y_true"):
                    with results_lock:
                        all_y_true.extend(result["y_true"])
                        all_y_probs.extend(result["y_probs"])
                        per_worker_val[idx] = (result["y_true"], result["y_probs"])
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Calibrazione soglia: worker '{w_name}' "
                      f"non ha fornito il validation set ({e}). Escluso dal pool.")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        threads = [
            threading.Thread(target=_probe, args=(w, info, self._infer_worker_index(w, i)))
            for i, (w, info) in enumerate(available_workers.items(), start=1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=90)

        MIN_VAL_SAMPLES = 30  # da tarare, stesso ordine di grandezza di MIN_SAMPLES_PER_CLASS in _execute_inference_step
        if len(all_y_true) < MIN_VAL_SAMPLES or len(set(all_y_true)) < 2:
            print(f"[{self.orchestrator_name}] [WARN] Validation set federato insufficiente per la "
                  f"calibrazione ({len(all_y_true)} campioni, classi presenti: {set(all_y_true)}). "
                  f"Nessuna soglia federata calcolata: l'inferenza ricadrà sulla soglia della baseline.")
            return None

        precisions, recalls, thresholds = precision_recall_curve(all_y_true, all_y_probs)
        denom = precisions + recalls
        f1_scores = np.where(denom == 0, 0.0, 2 * precisions * recalls / np.where(denom == 0, 1, denom))
        best_idx = int(np.nanargmax(f1_scores[:-1]))  # l'ultimo punto della curva non ha una soglia associata
        best_threshold = float(thresholds[best_idx])

        print(f"[{self.orchestrator_name}] [CALIBRAZIONE SOGLIA] Soglia federata calibrata su "
              f"{len(all_y_true)} campioni di validation (pool cross-worker): {best_threshold:.4f} "
              f"(F1={f1_scores[best_idx]:.4f}, precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f}).")

        try:
            threshold_path = self._resolve_threshold_path(job_id)
            self.checkpoint_dao.save(threshold_path, {
                "decision_threshold": best_threshold,
                "n_val_samples": len(all_y_true),
                "f1_score": float(f1_scores[best_idx]),
                "precision": float(precisions[best_idx]),
                "recall": float(recalls[best_idx]),
            })
        except Exception as e_save:
            print(f"[{self.orchestrator_name}] [WARN] Soglia federata calcolata ma non salvata "
                  f"({e_save}): l'inferenza ricadrà sulla soglia della baseline.")
            return None

        # SOGLIE PER-WORKER: con partizionamento non-IID (es. 'by_day') la
        # soglia unica calibrata sul pool cross-worker può essere lontana
        # dall'ottimo per un singolo worker la cui distribuzione locale delle
        # probabilità è sistematicamente diversa dal resto (vedi analisi sui
        # worker 5/10: non un problema di bilanciamento per-albero, ma di
        # soglia globale applicata a una distribuzione locale spostata).
        # Calcoliamo quindi, in aggiunta a quella globale, una soglia
        # ottimizzata per F1 SOLO sui dati di validation locali di ciascun
        # worker (stessa curva precision-recall, stesso criterio, ma sample
        # ristretto a quel worker). Se un worker non ha abbastanza campioni
        # locali (stesso MIN_VAL_SAMPLES della soglia globale) o ha una sola
        # classe nel proprio validation set, non gli si assegna una soglia
        # propria: in inferenza ricadrà sulla soglia globale per quel worker
        # (vedi uso di questo dizionario in _execute_inference_step).
        #
        # GUARDIA AUC (aggiunta dopo il caso worker 5/10): un modello globale
        # può semplicemente non avere segnale discriminante sui dati locali
        # di un worker (giorno con pattern di traffico/attacco strutturalmente
        # diverso da quelli su cui il resto dell'ensemble è stato allenato --
        # concept shift, non un problema di soglia). In quel caso l'AUC locale
        # è vicino a 0.5 (ranking ~casuale), e la soglia "ottima per F1" degenera
        # matematicamente nel predire sempre la classe locale maggioritaria: su
        # worker 5 (91% classe 1) e worker 10 (68% classe 1) questo significava
        # "predici sempre attacco", con F1 aggregato migliore sulla carta ma un
        # sistema operativamente inutile su quei giorni specifici (falsi
        # positivi costanti). Sotto MIN_LOCAL_AUC non assegniamo una soglia
        # individuale: quel worker ricade sulla soglia globale, che pur non
        # essendo ottima per lui almeno non sfrutta ciecamente lo sbilanciamento
        # locale delle classi.
        MIN_LOCAL_AUC = 0.65
        per_worker_thresholds = {}
        for w_idx, (y_true_w, y_probs_w) in per_worker_val.items():
            if len(y_true_w) < MIN_VAL_SAMPLES or len(set(y_true_w)) < 2:
                continue
            try:
                local_auc = roc_auc_score(y_true_w, y_probs_w)
                if local_auc < MIN_LOCAL_AUC:
                    print(f"[{self.orchestrator_name}] [CALIBRAZIONE SOGLIA PER-WORKER] Worker {w_idx}: "
                          f"AUC locale {local_auc:.4f} sotto la soglia minima ({MIN_LOCAL_AUC}) -- probabile "
                          f"concept shift rispetto al resto dell'ensemble, non un problema di soglia. "
                          f"Nessuna soglia individuale assegnata: userà quella globale.")
                    continue

                p_w, r_w, t_w = precision_recall_curve(y_true_w, y_probs_w)
                denom_w = p_w + r_w
                f1_w = np.where(denom_w == 0, 0.0, 2 * p_w * r_w / np.where(denom_w == 0, 1, denom_w))
                best_w = int(np.nanargmax(f1_w[:-1]))
                per_worker_thresholds[w_idx] = {
                    "decision_threshold": float(t_w[best_w]),
                    "n_val_samples": len(y_true_w),
                    "f1_score": float(f1_w[best_w]),
                    "precision": float(p_w[best_w]),
                    "recall": float(r_w[best_w]),
                    "local_auc": float(local_auc),
                }
            except Exception as e_w:
                print(f"[{self.orchestrator_name}] [WARN] Calibrazione soglia per-worker fallita per "
                      f"il worker {w_idx} ({e_w}): quel worker userà la soglia globale in inferenza.")

        if per_worker_thresholds:
            try:
                self.checkpoint_dao.save(self._resolve_per_worker_threshold_path(job_id), per_worker_thresholds)
                riepilogo = ", ".join(
                    f"{idx}={v['decision_threshold']:.4f}" for idx, v in sorted(per_worker_thresholds.items())
                )
                print(f"[{self.orchestrator_name}] [CALIBRAZIONE SOGLIA PER-WORKER] Soglie individuali "
                      f"calcolate per {len(per_worker_thresholds)}/{len(per_worker_val)} worker: {riepilogo}. "
                      f"Gli altri useranno la soglia globale ({best_threshold:.4f}).")
            except Exception as e_save_w:
                print(f"[{self.orchestrator_name}] [WARN] Soglie per-worker calcolate ma non salvate "
                      f"({e_save_w}): l'inferenza userà solo la soglia globale per tutti i worker.")

        return best_threshold

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int, alberi_reali: list = None):
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

        if alberi_reali is not None and len(alberi_reali) > 0:
            try:
                # Sostituzione integrale dello stato: si azzera e si riscrive come
                # parte 0. Percorso oggi mai esercitato — BaseOrchestrator chiama
                # _save_checkpoint senza 'alberi_reali' — ma va tenuto coerente
                # col formato a parti, altrimenti reintrodurrebbe un monolitico.
                self._purge_trees_checkpoint(job_id)
                self._persist_trees_delta(job_id, alberi_reali, 0, 0)
                checkpoint_trees_path = self._resolve_trees_checkpoint_path(job_id)
                print(f"[{self.orchestrator_name}] Checkpoint alberi salvato in {checkpoint_trees_path}.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE CHECKPOINT] Impossibile salvare checkpoint alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        super()._clean_checkpoint(job_id)
        self._trees_cache.pop(job_id, None)
        try:
            # Rimuove tutte le parti incrementali oltre all'eventuale monolitico.
            self._purge_trees_checkpoint(job_id)
            print(f"[{self.orchestrator_name}] Checkpoint alberi rimosso per il job {job_id}.")
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

    def _resolve_model_meta_path(self, job_id: str) -> str:
        """Path dei metadati leggeri (conteggio alberi, classi) associati al
        modello globale: evita che _execute_inference_step debba deserializzare
        l'intero blob del modello (fino a 1+ GB con alberi non regolarizzati)
        solo per leggere questi due valori. Stessa sotto-cartella/convenzione
        di _resolve_model_path, suffisso 'model_meta_' invece di 'model_'."""
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/federated/model_meta_{job_id}.pkl"
        return os.path.join("./saved_models", f"model_meta_{job_id}.pkl")

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

    def _get_inference_marker_path(self, job_id: str) -> str:
        """
        Path del marcatore 'inferenza avviata'. Solo per ambiente 'local': è un
        segnale di osservabilità pensato per monitor/test sullo stesso filesystem
        (via bind mount in Docker), non un artefatto di stato distribuito.
        """
        return f"./.local_storage/inference_started_{job_id}.marker"

    def _mark_inference_started(self, job_id: str):
        """Crea il marcatore leggero che segnala l'avvio dell'inferenza federata."""
        if self.environment != "local" or not job_id:
            return
        try:
            marker = self._get_inference_marker_path(job_id)
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        except Exception as e:
            print(f"[{self.orchestrator_name}] [WARN] Impossibile creare il marcatore di inferenza per {job_id[:8]}: {e}")

    def _clear_inference_started(self, job_id: str):
        """Rimuove il marcatore a inferenza conclusa (o fallita). Idempotente."""
        if self.environment != "local" or not job_id:
            return
        try:
            marker = self._get_inference_marker_path(job_id)
            if os.path.exists(marker):
                os.remove(marker)
        except Exception:
            pass

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
        """Alias storico: delega al metodo condiviso in BaseOrchestrator
        (read_selected_features_from_config), riusato ora anche da
        centralized.py. Mantenuto per non rompere le chiamate esistenti in
        questo file."""
        return self.read_selected_features_from_config(dataset_type)

if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()