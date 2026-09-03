from abc import ABC, abstractmethod
import fcntl
import json
import os
import statistics
import sys
import time
import numpy as np
import rpyc
import signal
import threading
from src.dataset.metrics_dao import MetricsDAOFactory
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, f1_score, roc_auc_score
from src.shared.config import SystemConfig
from src.shared.factory import get_aws_services
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.binding.taskregistry import TaskRegistry
from src.shared.mock_aws.dynamodb.dynamodb_factory import DynamoDBFactory

BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")
# Tabella DynamoDB per il sidecar dei metadati di job (vedi _save_job_meta):
# equivalente AWS del file JSON usato in locale, necessaria perché
# _perform_active_recovery possa ricostruire un payload FEDELE all'originale
# (iperparametri, tipo/percorso dataset, partizionamento federato) dopo un
# failover dell'orchestratore, invece di ricadere sui default. Va creata a
# parte (stesso schema chiave delle altre tabelle usate qui, es. WorkerTasks/
# OrchestratorLocks): non viene provisionata da questo codice.
JOB_META_TABLE = "JobMetadata"

# --------------------------------------------------------------------------- #
# Normalizzazione dei tipi numerici dei metadati di job                        #
# --------------------------------------------------------------------------- #
#
# PERCHÉ ESISTE QUESTO BLOCCO
# DynamoDB non ha un tipo "float": AwsDynamoDB._to_dynamo converte i float in
# Decimal in scrittura, e AwsDynamoDB._from_dynamo li riconverte in lettura con
#
#     return int(value) if value % 1 == 0 else float(value)
#
# cioè OGNI Decimal con parte frazionaria nulla torna come int. È la scelta
# giusta per la stragrande maggioranza dei campi che passano di lì
# (alberi_addestrati, expires_at, timestamp, n_estimators: devono essere int, e
# restituirli come float romperebbe range(), gli indici e i confronti), quindi
# NON va cambiata in _from_dynamo.
#
# Per un pugno di iperparametri, però, int e float non sono lo stesso numero
# scritto in due modi: sono due SEMANTICHE DIVERSE in scikit-learn.
#
#     max_samples=1.0  (float) -> ogni albero campiona il 100% del training set
#     max_samples=1    (int)   -> ogni albero campiona UN SINGOLO record
#
# Il manifesto della baseline dichiara max_samples: 1.0. Senza questa
# normalizzazione, un job ripreso da _perform_active_recovery dopo un failover
# ricostruirebbe la foresta con un albero per campione: nessuna eccezione,
# nessun log anomalo, solo metriche prive di senso dopo il failover.
#
# PERCHÉ QUI E NON NEL MODELLO PYDANTIC
# Hyperparameters.check_max_samples farebbe già la coercizione int -> float, ma
# il percorso di recovery non passa dal modello: _perform_active_recovery mette
# job_meta.get("hyperparameters", {}) grezzo nel payload e l'orchestratore lo
# legge con hp.get("max_samples") (vedi CentralizedOrchestrator, ramo che
# inoltra bootstrap/max_samples a train_subset_forest). La normalizzazione va
# quindi fatta al confine di lettura dei metadati, dove i tipi attesi sono noti.
#
# In locale il round-trip JSON preserva già 1.0 come float: là la funzione è un
# no-op, ed è applicata comunque per non avere due comportamenti diversi fra i
# due ambienti.

# Iperparametri che devono restare float anche quando il valore è numericamente
# intero. Nota su max_features: scikit-learn accetterebbe anche un int (= numero
# assoluto di feature), ma il modello Hyperparameters lo dichiara
# Optional[Union[str, float]], quindi in questo sistema un max_features numerico
# è SEMPRE una frazione. Restano ovviamente intatti i valori stringa
# ('sqrt', 'log2'), che non sono numeri.
_JOB_META_FLOAT_HP_KEYS = ("max_samples", "max_features")

# Campi float di primo livello nei metadati. partition_alpha è l'iperparametro
# di eterogeneità della partizione Dirichlet: numericamente 1 e 1.0 sono
# equivalenti per numpy, ma tenerlo float mantiene coerente ciò che finisce
# nelle metriche dell'esperimento.
_JOB_META_FLOAT_KEYS = ("partition_alpha",)


def _as_float_if_integral(value):
    """
    Riporta a float un valore che deve esserlo ma è tornato int dal round-trip
    DynamoDB. Lascia intatto tutto il resto: None, stringhe ('sqrt'), float già
    corretti. I bool sono esclusi esplicitamente perché in Python bool è
    sottoclasse di int, e True diventerebbe 1.0.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return float(value)
    return value


def normalize_job_meta_numerics(meta: dict) -> dict:
    """
    Restituisce una copia di 'meta' con i campi numerici riportati al tipo
    atteso. Difensiva per costruzione: un meta vuoto, parziale o con
    'hyperparameters' di tipo inatteso attraversa la funzione senza modifiche,
    perché un sidecar malformato non deve impedire il recovery — deve solo non
    peggiorarlo.
    """
    if not isinstance(meta, dict) or not meta:
        return meta

    normalized = dict(meta)

    for key in _JOB_META_FLOAT_KEYS:
        if key in normalized:
            normalized[key] = _as_float_if_integral(normalized[key])

    hp = normalized.get("hyperparameters")
    if isinstance(hp, dict):
        hp_normalized = dict(hp)
        for key in _JOB_META_FLOAT_HP_KEYS:
            if key in hp_normalized:
                hp_normalized[key] = _as_float_if_integral(hp_normalized[key])
        normalized["hyperparameters"] = hp_normalized

    return normalized


def env_timeout_seconds(var_name: str, default: int) -> int:
    """
    Legge un timeout espresso in secondi da variabile d'ambiente, restituendo
    'default' se la variabile è assente, vuota o non interpretabile.

    PERCHÉ NON BASTA int(os.environ.get(NOME, default))
    deploy.sh, quando la chiave non è presente nel .env, ripiega su valori
    scritti CON il suffisso: DETECTED_RPC_SYNC_TIMEOUT="${ENV_RPC_SYNC_TIMEOUT:-1800s}".
    Quel valore finisce tale e quale nella task definition ECS, e int("1800s")
    solleva ValueError — a livello di modulo, quindi PRIMA che qualunque
    try/except applicativo possa intercettarlo: il container morirebbe
    all'import con uno stack trace incomprensibile, e solo sulle macchine il
    cui .env non dichiara quella chiave.

    Qui il suffisso 's' viene tollerato, e qualunque altro valore malformato
    produce un WARN esplicito più il default, invece di un crash. Un timeout
    sbagliato è un problema; un orchestratore che non parte è peggio.
    """
    raw = os.environ.get(var_name)
    if raw is None:
        return default

    cleaned = str(raw).strip()
    if not cleaned:
        return default
    if cleaned[-1] in ("s", "S"):
        cleaned = cleaned[:-1].strip()

    try:
        value = int(cleaned)
    except (TypeError, ValueError):
        print(f"[CONFIG] [WARN] {var_name}='{raw}' non è un numero di secondi valido: "
              f"uso il default {default}s.")
        return default

    if value <= 0:
        print(f"[CONFIG] [WARN] {var_name}='{raw}' non è positivo: uso il default {default}s.")
        return default
    return value


class MessageOwnershipLostError(Exception):
    """Eccezione personalizzata per indicare la perdita di ownership del messaggio SQS."""
    pass


class BaseOrchestrator(ABC):
    def __init__(self, orchestrator_name: str, queue_name: str):
        self.cfg = SystemConfig()
        self.environment = self.cfg.env
        self._stop_heartbeat = threading.Event()
        self.orchestrator_name = orchestrator_name
        self.queue_name = queue_name
        self.connessioni_attive = []
        self.connessioni_lock = threading.Lock()
        
        try:
            self.sqs_queue, self.state_manager = get_aws_services(self.environment)
        except Exception as e:
            print(f"[{self.orchestrator_name.upper()}] Errore inizializzazione servizi: {e}")
            sys.exit(1)

    def _measure_rpc_ping_stats(self, num_probes: int = 5) -> dict:
        """
        Misura la latenza RPC PURA (round-trip di exposed_ping, senza alcun
        ETL/training coinvolto) verso ogni worker attualmente disponibile.

        A differenza delle misure basate su _execute_training_step (che
        includono sempre l'intero ETL prima di contattare il worker), qui il
        tempo cronometrato è ESCLUSIVAMENTE quello del round-trip RPyC:
        apertura già esclusa dal timing (fuori dal ciclo), viene misurato
        solo request/response di root.ping() per num_probes volte per worker.

        Condiviso da CentralizedOrchestrator e FederatedOrchestrator: usa lo
        stesso pattern di connessione (ServiceRegistry + rpyc.connect +
        self.connessioni_lock/self.connessioni_attive) già impiegato nei
        rispettivi _execute_training_step.

        Ritorna un dizionario con le statistiche aggregate su TUTTI i worker
        e le probe, più la media per singolo worker (utile per individuare
        eventuali worker anomali/più lenti degli altri).
        """
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        per_worker_avg_ms = {}
        samples_ms = []

        if not available_workers:
            print(f"[{self.orchestrator_name}] [PING] Nessun worker disponibile per la misura di latenza pura.")
            return {
                "samples_ms": [], "min_ms": None, "max_ms": None,
                "avg_ms": None, "median_ms": None, "per_worker_avg_ms": {},
            }

        for w_name, w_info in available_workers.items():
            worker_conn = None
            worker_samples = []
            try:
                worker_conn = rpyc.connect(
                    w_info["host"], w_info["port"],
                    config={'allow_pickle': True, 'sync_request_timeout': 30, 'keepalive': True}
                )
                with self.connessioni_lock:
                    self.connessioni_attive.append(worker_conn)

                for _ in range(num_probes):
                    t0 = time.perf_counter()
                    worker_conn.root.ping()
                    worker_samples.append((time.perf_counter() - t0) * 1000)

            except Exception as e:
                print(f"[{self.orchestrator_name}] [PING] Errore contattando {w_name}: {e}")
            finally:
                if worker_conn:
                    with self.connessioni_lock:
                        if worker_conn in self.connessioni_attive:
                            self.connessioni_attive.remove(worker_conn)
                    try:
                        worker_conn.close()
                    except Exception:
                        pass

            if worker_samples:
                per_worker_avg_ms[w_name] = round(statistics.mean(worker_samples), 3)
                samples_ms.extend(worker_samples)

        if not samples_ms:
            return {
                "samples_ms": [], "min_ms": None, "max_ms": None,
                "avg_ms": None, "median_ms": None, "per_worker_avg_ms": per_worker_avg_ms,
            }

        return {
            "samples_ms": [round(s, 3) for s in samples_ms],
            "min_ms": round(min(samples_ms), 3),
            "max_ms": round(max(samples_ms), 3),
            "avg_ms": round(statistics.mean(samples_ms), 3),
            "median_ms": round(statistics.median(samples_ms), 3),
            "per_worker_avg_ms": per_worker_avg_ms,
        }

    def _track_task(self, task_id, job_id: str, worker_name: str, status: str):

        try:
            db = DynamoDBFactory.get_db(self.environment)
            db.put_item("WorkerTasks", f"{job_id}_{task_id}", {
                "job_id": job_id,
                "task_id": task_id,
                "worker_name": worker_name,
                "status": status,
                "update_at": int(time.time())
            })
        except Exception as e:
            print(f"[{self.orchestrator_name}] Errore tracciamento task {task_id} per job {job_id[:8]}: {e}")

    def _get_lock_key(self) -> str:
        return "global_orchestrator_leader_lock"
    
    def _try_acquire_leadership(self, ttl: int = 180) -> bool:
            
        lock_key = self._get_lock_key()

        if self.environment == "local":
            lock_dir = "./.local_storage"
            lock_path = os.path.join(lock_dir, f"{lock_key}.json")
            mutex_path = os.path.join(lock_dir, f"{lock_key}.mutex")
            now = time.time()
            os.makedirs(lock_dir, exist_ok=True)

            with open(mutex_path, "a") as mutex:
                fcntl.flock(mutex, fcntl.LOCK_EX)
                try:
                    if os.path.exists(lock_path):
                        try:
                            with open(lock_path, "r", encoding="utf-8") as f:
                                lock_data = json.load(f)
                            owner = lock_data.get("leader")
                            timestamp = lock_data.get("timestamp", 0)
                            # Lock valido e di qualcun altro → standby
                            if owner != self.orchestrator_name and (now - timestamp) < ttl:
                                return False
                        except (json.JSONDecodeError, KeyError, ValueError):
                            print(f"[{self.orchestrator_name}] Lock corrotto, tento sovrascrittura...")

                    # Siamo dentro il mutex: scrittura diretta, niente .tmp necessario
                    with open(lock_path, "w", encoding="utf-8") as f:
                        json.dump({"leader": self.orchestrator_name, "timestamp": now}, f, indent=2)
                    return True

                except Exception as e:
                    print(f"[{self.orchestrator_name}] Errore acquisizione lock: {e}")
                    return False
                finally:
                    fcntl.flock(mutex, fcntl.LOCK_UN)
        else:
            if not hasattr(self.state_manager, "acquire_global_lock"):
                print(f"[{self.orchestrator_name}] [WARN] state_manager non supporta i lock: "
                    f"leadership assegnata senza coordinamento (comportamento degradato).")
                return True

            try:
                return self.state_manager.acquire_global_lock(lock_key, self.orchestrator_name, ttl=ttl)
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE] Acquisizione lock fallita: {e}")
                return False
    
    def _refresh_leadership_lock(self):
        lock_key = self._get_lock_key()

        if self.environment == "local":
            lock_dir = "./.local_storage"
            lock_path = os.path.join(lock_dir, f"{lock_key}.json")
            mutex_path = os.path.join(lock_dir, f"{lock_key}.mutex")

            os.makedirs(lock_dir, exist_ok=True)

            with open(mutex_path, "a") as mutex:
                fcntl.flock(mutex, fcntl.LOCK_EX)
                try:
                    if os.path.exists(lock_path):
                        try:
                            with open(lock_path, "r", encoding="utf-8") as f:
                                lock_data = json.load(f)
                            if lock_data.get("leader") != self.orchestrator_name:
                                print(f"[{self.orchestrator_name}] Refresh ignorato: non sono più il leader.")
                                return False
                        except (json.JSONDecodeError, KeyError):
                            pass

                    with open(lock_path, "w", encoding="utf-8") as f:
                        json.dump({"leader": self.orchestrator_name, "timestamp": time.time()}, f, indent=2)
                    return True
                except Exception as e:
                    print(f"[{self.orchestrator_name}] Errore nel rinnovo del lock: {e}")
                finally:
                    fcntl.flock(mutex, fcntl.LOCK_UN)
        else:
            try:
                return bool(self.state_manager.refresh_global_lock(lock_key, self.orchestrator_name, ttl=180))
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE] Refresh lock fallito: {e}")
                return False

    def _release_leadership(self):

        lock_key = self._get_lock_key()

        if self.environment == "local":
            lock_dir = "./.local_storage"
            lock_path = os.path.join(lock_dir, f"{lock_key}.json")
            mutex_path = os.path.join(lock_dir, f"{lock_key}.mutex")

            os.makedirs(lock_dir, exist_ok=True)

            with open(mutex_path, "a") as mutex:
                fcntl.flock(mutex, fcntl.LOCK_EX)
                try:
                    if os.path.exists(lock_path):
                        with open(lock_path, "r", encoding="utf-8") as f:
                            lock_data = json.load(f)
                        if lock_data.get("leader") == self.orchestrator_name:
                            os.remove(lock_path)
                            print(f"[{self.orchestrator_name}] Lock di leadership globale rilasciato.")
                except Exception:
                    pass
                finally:
                    fcntl.flock(mutex, fcntl.LOCK_UN)
        else:
            try:
                self.state_manager.release_global_lock(lock_key, self.orchestrator_name)
            except Exception:
                pass

    def _cleanup_dead_workers(self):
        try:
            expired = ServiceRegistry.get_expired_workers()
        except Exception as e:
            print(f"[{self.orchestrator_name}] [WARN] Impossibile leggere i worker scaduti: {e}")
            return
 
        for worker_name, info in expired.items():
            print(f"[{self.orchestrator_name}] [CLEANUP] Worker '{worker_name}' scaduto "
                  f"({info['seconds_since_heartbeat']}s senza heartbeat). Deregistrazione in corso...")
 
            try:
                pending_tasks = TaskRegistry.get_tasks_by_worker(worker_name)
                orphaned = [t for t in pending_tasks if t.get("status") not in ("COMPLETED", "FAILED")]
                if orphaned:
                    print(f"[{self.orchestrator_name}] [CLEANUP] Worker '{worker_name}' aveva "
                          f"{len(orphaned)} task non conclusi al momento della scadenza: "
                          f"{[(t.get('job_id'), t.get('status')) for t in orphaned]}")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Impossibile leggere WorkerTasks per '{worker_name}': {e}")
 
            ServiceRegistry.deregister_worker(worker_name)


    def _heartbeat_loop(self, stop_event: threading.Event,leadership_lost_event:threading.Event, interval: int = 10):
        """Invia heartbeat di rete e tiene in vita il lock di leadership ogni 10 secondi."""
        while not stop_event.is_set():
            try:
                
                ServiceRegistry.update_orchestrator_heartbeat(self.orchestrator_name)
                if not self._refresh_leadership_lock():
                    print(f"[{self.orchestrator_name}] [LEADERSHIP LOST] Il lock non è più nostro. Rientro in standby.")
                    leadership_lost_event.set()
                    return
                self._cleanup_dead_workers()
            except Exception as e:
                print(f"[{self.orchestrator_name}] Errore durante l'aggiornamento del heartbeat/lock: {e}")

            for _ in range(interval):
                if stop_event.is_set():
                    break
                time.sleep(1)

    def _job_lease_heartbeat_loop(self, job_id: str, stop_event: threading.Event,
                               lease_lost_event: threading.Event,
                               lease_seconds: int = 300, interval: int = 60):
        """Rinnova periodicamente la lease del job, indipendentemente dalla durata
        dello step di training in corso (che può superare abbondantemente lease_seconds
        a causa dei timeout RPC verso i worker)."""
        while not stop_event.is_set():
            for _ in range(interval):
                if stop_event.is_set():
                    return
                time.sleep(1)
            try:
                if not self.state_manager.try_claim_job(job_id, self.orchestrator_name, lease_seconds=lease_seconds):
                    print(f"[{self.orchestrator_name}] [JOB-LEASE] Lease persa per il job {job_id[:8]}!")
                    lease_lost_event.set()
                    return
            except Exception as e:
                print(f"[{self.orchestrator_name}] [JOB-LEASE-ERROR] {e}")
                
    def _visibility_heartbeat_loop(self, receipt_handle: str, stop_event: threading.Event, ownership_lost_event: threading.Event, timeout_extension: int = 180, interval: int = 60):
        """
        Invia periodicamente un comando a SQS per estendere l'invisibilità del messaggio
        correntemente in elaborazione, finché l'evento stop_event non viene settato.
        """
        print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT] Thread avviato per il messaggio corrente.")

        first_wait = interval // 2  # primo rinnovo anticipato di sicurezza
        while not stop_event.is_set():
            for _ in range(first_wait if first_wait else interval):
                if stop_event.is_set():
                    return
                time.sleep(1)
            first_wait = interval
            try:
                self.sqs_queue.change_message_visibility(
                    queue_name=self.queue_name,
                    receipt_handle=receipt_handle,
                    visibility_timeout=timeout_extension
                )
            except Exception as e:
                print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT-ERROR] {e}")
                ownership_lost_event.set()
                return  # inutile continuare a girare, l'ownership è persa

    def start(self, stop_event: threading.Event = None):
        """
        Metodo Template: gestisce l'intero ciclo di vita del polling e del failover.

        stop_event: opzionale, retrocompatibile (default None = comportamento
        storico invariato, usato dall'entrypoint reale in produzione). Se
        fornito, viene controllato ad ogni iterazione del loop principale:
        serve SOLO a chi avvia questo metodo in un thread in background (es.
        gli scenari di test di failover, che istanziano un Leader/Standby
        "usa e getta" per la durata del singolo scenario) per poterlo fermare
        pulitamente al termine — altrimenti, dato che qui SIGTERM/KeyboardInterrupt
        sono disabilitati fuori dal main thread (vedi sotto), il loop
        resterebbe attivo per sempre, anche dopo che lo scenario è concluso.
        """
        print("=====================================================")
        print(f"  {self.orchestrator_name.upper()} IN ASCOLTO ({self.environment.upper()})...")
        print("=====================================================\n")

        ServiceRegistry.register_orchestrator(self.orchestrator_name)
        def _handle_sigterm(signum, frame):
            raise KeyboardInterrupt()

      
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _handle_sigterm)
        else:
            print(f"[{self.orchestrator_name}] [WARN] start() eseguito fuori dal main thread: "
                  f"gestione di SIGTERM disabilitata per questa istanza.")
        is_leader = False
        
        self.hb_thread = None
        leadership_lost_event = threading.Event()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    print(f"[{self.orchestrator_name}] [STOP] Richiesta di arresto ricevuta (stop_event): "
                          f"uscita pulita dal loop di polling.")
                    break
                if leadership_lost_event.is_set():
                    print(f"[{self.orchestrator_name}] [DOWNGRADE] Rientro in standby, riprovo l'acquisizione.")
                    is_leader = False
                    leadership_lost_event.clear()
                if not is_leader:
                    is_leader = self._try_acquire_leadership()
                    if not is_leader:
                        print(f"[{self.orchestrator_name}] [STANDBY] Un altro orchestratore è Active. In attesa di fault... (Sleep 5s)")
                        time.sleep(5)
                        continue
                    else:
                        print(f"\n[{self.orchestrator_name}] [ACTIVE] !!! LEADERSHIP ACQUISITA !!!")
                        print(f"[{self.orchestrator_name}] Avvio dell'heartbeat thread e attivazione del polling sulla coda: '{self.queue_name}'\n")
                        # Avviamo il thread di heartbeat SOLO dopo aver conquistato la leadership
                        self.hb_thread = threading.Thread(target=self._heartbeat_loop, args=(self._stop_heartbeat,leadership_lost_event), daemon=True)
                        self.hb_thread.start()

                        try:
                            self._perform_active_recovery()
                        except Exception as recovery_error:
                            print(f"[{self.orchestrator_name}] [ERRORE RECOVERY] Il ripristino attivo è fallito: {recovery_error}")
                            import traceback
                            traceback.print_exc()
                                    
                try:
                    sqs_response = self.sqs_queue.receive_message(queue_name=self.queue_name, visibility_timeout=180)

                    if not sqs_response:
                        time.sleep(5)
                        continue

                    receipt_handle = sqs_response["ReceiptHandle"]
                    payload = sqs_response["Body"]
                    
                    self._process_job(payload, receipt_handle)

                except Exception as infra_error:
                    print(f"\n[{self.orchestrator_name}] [ERRORE INFRASTRUTTURALE]: {infra_error}")
                    import traceback          
                    traceback.print_exc()
                    time.sleep(10)
        except KeyboardInterrupt:
            print(f"\n[-] Interruzione manuale intercettata sull'orchestrattore {self.orchestrator_name}")
        finally:
            print(f"[*] Chiusura dei servizi in corso per {self.orchestrator_name}...")
            if is_leader:
                self._stop_heartbeat.set()
                if self.hb_thread:
                    self.hb_thread.join(timeout=2)
                self._release_leadership()
            ServiceRegistry.deregister_orchestrator(self.orchestrator_name)
            print(f"[*] Orchestratore rimosso correttamente dalla rete.")

    def _process_job(self, payload: dict, receipt_handle: str):

        """Logica di instradamento del lavoro in base al tipo di richiesta."""
        job_id = payload.get("job_id")
        request_type = payload.get("request_type", "TRAINING").upper()
        stop_visibility = threading.Event()
        ownership_lost_event = threading.Event()
        stop_job_lease = threading.Event()
        job_lease_lost_event = threading.Event()
        visibility_thread = None
        if receipt_handle:
            visibility_thread = threading.Thread(
                target=self._visibility_heartbeat_loop,
                args=(receipt_handle, stop_visibility, ownership_lost_event),
                daemon=True
            )
            visibility_thread.start()
        job_lease_thread = threading.Thread(
            target=self._job_lease_heartbeat_loop,
            args=(job_id, stop_job_lease, job_lease_lost_event),
            kwargs={"lease_seconds": 300, "interval": 60},
            daemon=True
        )
        job_lease_thread.start()
        try:
           
            self._save_job_meta(job_id, payload)

            if request_type == "INFERENCE":
                print(f"\n[{self.orchestrator_name}] Ricevuta richiesta di INFERENZA per il Job ID: {job_id[:8]}...")
                try:
                    self._execute_inference_step(payload)
                    if receipt_handle:
                        self.sqs_queue.delete_message(receipt_handle)
                    print(f"[{self.orchestrator_name}] Inferenza per Job {job_id[:8]} completata con successo.")
                except Exception as inf_error:
                    print(f"[{self.orchestrator_name}] [ERRORE DURANTE INFERENZA]: {inf_error}")
                    import traceback
                    traceback.print_exc()
                return

            status = self.state_manager.get_job_status(job_id)
            if status == "COMPLETED":
                print(f"[INFO] Job {job_id[:8]} già completato. Ignoro messaggio duplicato.")
                if receipt_handle:
                    self.sqs_queue.delete_message(receipt_handle)
                return 
                
            hp = payload.get("hyperparameters", {})
            existing_state = self.state_manager.obtain_request(job_id)
            retries = 0
            base_random_state = 123
            alberi_gia_fatti = 0

            if existing_state:
                item_data = existing_state.get("Item", existing_state)
                current_status = item_data.get("status")
                retries = item_data.get("retries", 0)
                base_random_state = item_data.get("base_random_state", 123)
                
                if current_status == "PROCESSING":
                    print(f"[{self.orchestrator_name}] [FAILOVER DETECTED] Riprendo il lavoro del nodo fallito.")
                    alberi_gia_fatti = self._load_checkpoint(job_id, item_data)
                    retries += 1
                elif current_status == "COMPLETED":
                    print(f"[{self.orchestrator_name}] Job già completato. Scarto.")
                    if receipt_handle:
                        self.sqs_queue.delete_message(receipt_handle)
                    return
            
            if not self.state_manager.try_claim_job(job_id, self.orchestrator_name, lease_seconds=300):
                print(f"[{self.orchestrator_name}] [ABORT] Job {job_id[:8]} già in possesso di un altro Orchestrator.")
                return
            self.state_manager.update_request_status(
                job_id=job_id, 
                status="PROCESSING", 
                orchestrator_id=self.orchestrator_name, 
                retries=retries,
                base_random_state=base_random_state,
                alberi_addestrati=alberi_gia_fatti
            )

            start_dist = time.perf_counter()

            try:
                alberi_totali = hp.get("n_estimators", 100)
                num_worker_attuali  = max(1, self._get_active_worker_count())
                step_alberi = max(20, num_worker_attuali * 10)  # Step dinamico basato sul numero di worker attivi
                current_alberi = alberi_gia_fatti

                while current_alberi < alberi_totali:
                    if ownership_lost_event.is_set() or job_lease_lost_event.is_set():
                        raise MessageOwnershipLostError(
                            f"[{self.orchestrator_name}] Ownership persa a metà elaborazione: "
                            f"abort per evitare lavoro duplicato/corrotto."
                        )
                    if not self.state_manager.try_claim_job(job_id, self.orchestrator_name, lease_seconds=300):
                        raise MessageOwnershipLostError(
                            f"[{self.orchestrator_name}] Lease del job persa: un altro Orchestrator l'ha reclamata."
                        )
                    prossimo_target = min(current_alberi + step_alberi, alberi_totali)
                    payload["retries"] = retries
                    alberi_ottenuti = self._execute_training_step(payload, current_alberi, prossimo_target, base_random_state)

                    if alberi_ottenuti <= current_alberi:
                        print(f"[{self.orchestrator_name}] Nessun progresso nell'addestramento per Job {job_id[:8]}. Risorse insufficienti. In attesa di nuovi Worker...")
                        return
                    
                    current_alberi = alberi_ottenuti

                    self._save_checkpoint(job_id, current_alberi, retries, base_random_state)
            
                t_dist = time.perf_counter() - start_dist 
                self.state_manager.complete_request(job_id=job_id, orchestrator_id=self.orchestrator_name)
                try:
                    self.state_manager.release_job_lease(job_id, self.orchestrator_name)
                except Exception as e:
                    print(f"[{self.orchestrator_name}] [WARN] Impossibile rilasciare la job lease per {job_id[:8]}: {e}")
                if receipt_handle:
                    self.sqs_queue.delete_message(receipt_handle)
                self._clean_checkpoint(job_id)
                partitioning_info = {
                    "strategy": payload.get("partition_strategy", "iid"),
                    "alpha": payload.get("partition_alpha"),
                    "tree_allocation": payload.get("tree_allocation_strategy", "proportional"),
                }
                self._generate_performance_report(job_id, t_dist, current_alberi, partitioning_info=partitioning_info)
                
                print(f"[{self.orchestrator_name}] Job {job_id[:8]} completato con successo.")
            except MessageOwnershipLostError as ownership_error:
                print(f"[{self.orchestrator_name}] [ABORT] {ownership_error}")
                
            except Exception as eval_error:
                print(f"[{self.orchestrator_name}] [ERRORE APPLICATIVO]: {eval_error}")
                import traceback
                traceback.print_exc()
                self.state_manager.update_request_status(
                    job_id=job_id, 
                    status="FAILED", 
                    orchestrator_id="SYSTEM_ERR",
                    retries=retries,
                    base_random_state=base_random_state,
                    alberi_addestrati=current_alberi
                )
                try:
                    self.state_manager.release_job_lease(job_id, self.orchestrator_name)
                except Exception as e:
                    print(f"[{self.orchestrator_name}] [WARN] Impossibile rilasciare la job lease per {job_id[:8]}: {e}")
        except KeyboardInterrupt:
          
            print(f"[{self.orchestrator_name}] [INTERRUPTED] Interruzione durante l'elaborazione del Job {job_id[:8]}: rilascio la lease prima di terminare.")
            try:
                self.state_manager.release_job_lease(job_id, self.orchestrator_name)
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Impossibile rilasciare la job lease per {job_id[:8]}: {e}")
            raise  # ri-solleva per permettere allo shutdown esterno (in start()) di procedere normalmente
        finally:
            # Segnaliamo al thread di heartbeat di terminare
            stop_visibility.set()
            stop_job_lease.set()
            if visibility_thread:
                visibility_thread.join(timeout=2)
                print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT] Thread terminato per il messaggio corrente.")
            if job_lease_thread:
                job_lease_thread.join(timeout=2)
                print(f"[{self.orchestrator_name}] [JOB-LEASE] Thread terminato per il job corrente.")

    @abstractmethod
    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        pass

    @abstractmethod
    def _execute_inference_step(self, payload: dict):
        pass

    def _get_job_meta_path(self, job_id: str) -> str:
        return os.path.join("./.local_storage", "job_meta", f"job_meta_{job_id}.json")

    def _save_job_meta(self, job_id: str, payload: dict):
        """
        Persiste i metadati originali del job (dataset_path, dataset_type,
        hyperparameters, request_type, e — per il federato — partition_strategy/
        partition_alpha/tree_allocation_strategy). Lo state_manager (DynamoDB
        reale o mock) NON conserva questi campi: senza questo sidecar,
        _perform_active_recovery non potrebbe ricostruire un payload valido
        dopo un failover dell'orchestratore, e ripartirebbe con i default
        (es. n_estimators=100 invece del valore reale del manifesto).

        In locale: file JSON su disco (bind mount condiviso tra le istanze
        Docker). Su AWS: stessa idea via una piccola tabella DynamoDB
        dedicata (JOB_META_TABLE) — PRIMA questo metodo era no-op su AWS, il
        che faceva silenziosamente ricadere il recovery sui default proprio
        nell'ambiente (Fargate) dove il failover viene testato per davvero.
        """
        if not job_id:
            return
        meta = {
            "dataset_path": payload.get("dataset_path"),
            "dataset_type": payload.get("dataset_type"),
            "hyperparameters": payload.get("hyperparameters", {}),
            "request_type": payload.get("request_type", "TRAINING"),
            # Iperparametro dell'ESPERIMENTO federato (non del modello): senza
            # questi, un job federato ripreso dopo un failover perderebbe la
            # strategia di partizionamento/allocazione dichiarata nel job
            # originale, ripiegando sui default "iid"/"proportional" anche se
            # il job vero era, ad esempio, Dirichlet con allocazione equa —
            # etichettando le metriche del run recuperato in modo scorretto.
            "partition_strategy": payload.get("partition_strategy", "iid"),
            "partition_alpha": payload.get("partition_alpha"),
            "tree_allocation_strategy": payload.get("tree_allocation_strategy", "proportional"),
        }
        if self.environment == "local":
            try:
                meta_path = self._get_job_meta_path(job_id)
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Impossibile salvare i metadati del job {job_id[:8]}: {e}")
        else:
            try:
                db = DynamoDBFactory.get_db(self.environment)
                db.put_item(JOB_META_TABLE, job_id, meta)
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Impossibile salvare i metadati AWS (tabella "
                      f"'{JOB_META_TABLE}') del job {job_id[:8]}: {e}. Se la tabella non esiste ancora, "
                      f"il recovery su questo ambiente ricadrà sui default fino a quando non viene creata.")

    def _load_job_meta(self, job_id: str) -> dict:
        if self.environment == "local":
            meta_path = self._get_job_meta_path(job_id)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        # normalize_job_meta_numerics è un no-op sul ramo locale
                        # (JSON preserva i float): applicata comunque per non
                        # avere due comportamenti diversi fra locale e AWS.
                        return normalize_job_meta_numerics(json.load(f))
                except Exception as e:
                    print(f"[{self.orchestrator_name}] [WARN] Metadati del job {job_id[:8]} corrotti o illeggibili: {e}")
            else:
                print(f"[{self.orchestrator_name}] [WARN] Nessun sidecar di metadati trovato per il job {job_id[:8]}. Il recovery userà i default.")
            return {}
        else:
            try:
                db = DynamoDBFactory.get_db(self.environment)
                raw = db.get_item(JOB_META_TABLE, job_id)
                # Alcune implementazioni Dynamo (get_item "grezzo") avvolgono il
                # risultato in {"Item": {...}}; normalizziamo entrambi i casi.
                if isinstance(raw, dict) and "Item" in raw:
                    item = raw.get("Item") or {}
                else:
                    item = raw if isinstance(raw, dict) else {}
                if not item:
                    print(f"[{self.orchestrator_name}] [WARN] Nessun sidecar AWS di metadati trovato per il "
                          f"job {job_id[:8]} (tabella '{JOB_META_TABLE}'). Il recovery userà i default.")
                # QUI la normalizzazione è indispensabile: DynamoDB restituisce
                # Decimal, e AwsDynamoDB._from_dynamo trasforma in int ogni
                # Decimal a parte frazionaria nulla — max_samples: 1.0 tornerebbe
                # come 1, che per scikit-learn significa "un solo campione per
                # albero" invece di "il 100% del training set".
                item = normalize_job_meta_numerics(item)
                hp_check = item.get("hyperparameters") if isinstance(item, dict) else None
                if isinstance(hp_check, dict):
                    print(f"[{self.orchestrator_name}] [RECOVERY] Iperparametri recuperati dal sidecar: "
                          f"n_estimators={hp_check.get('n_estimators')}, max_depth={hp_check.get('max_depth')}, "
                          f"max_features={hp_check.get('max_features')}, criterion={hp_check.get('criterion')}, "
                          f"bootstrap={hp_check.get('bootstrap')}, max_samples={hp_check.get('max_samples')}")
                return item
            except Exception as e:
                print(f"[{self.orchestrator_name}] [WARN] Lettura metadati AWS del job {job_id[:8]} fallita: {e}. "
                      f"Il recovery userà i default.")
                return {}

    def _clean_job_meta(self, job_id: str):
        if self.environment == "local":
            meta_path = self._get_job_meta_path(job_id)
            if os.path.exists(meta_path):
                try:
                    os.remove(meta_path)
                except Exception:
                    pass
        else:
            try:
                db = DynamoDBFactory.get_db(self.environment)
                if hasattr(db, "delete_item"):
                    db.delete_item(JOB_META_TABLE, job_id)
            except Exception:
                pass  # cleanup best-effort: un record residuo non è dannoso

    def _load_checkpoint(self, job_id: str, existing_state: dict) -> int:
        db_val = existing_state.get("alberi_addestrati", 0)
        if self.environment == "local":
            path = os.path.join("./.local_storage", "checkpoints", f"checkpoint_{job_id}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f).get("alberi_addestrati", db_val)
        else:
            return db_val
        return db_val
    
    def _generate_performance_report(self, job_id: str, t_dist: float, alberi_addestrati: int = None,
                                      partitioning_info: dict = None):
        num_workers = self._get_active_worker_count()
        print("\n" + "═" * 75)
        print(f"  REPORT PRESTAZIONALE DISTRIBUITO - JOB {job_id[:8]}")
        print("═" * 75)
        print(f"  Tempo totale addestramento (T_dist):   {t_dist:.4f} s")
        print(f"  Worker utilizzati:                     {num_workers}")
        if alberi_addestrati is not None:
            print(f"  Alberi addestrati:                     {alberi_addestrati}")
        print("═" * 75 + "\n")

        # Persistenza delle metriche di training: stesso meccanismo gia usato per
        # l'inferenza, quindi finisce automaticamente in ./.local_storage/metrics
        # se environment == "local", o su s3://BUCKET_NAME/metrics/... se == "aws".
        mode = self.__class__.__name__.replace("Orchestrator", "").lower()
        metrics_entry = {
            "job_id": job_id,
            "mode": mode,
            "phase": "training",
            "timings": {"total_training_time": t_dist},
            "worker_count": num_workers,
            "alberi_addestrati": alberi_addestrati,
        }
        
        if partitioning_info is not None:
            metrics_entry["federated_partitioning"] = partitioning_info
        self._save_metrics(job_id, "training", metrics_entry)
    
    def _get_active_worker_count(self):
        workers = ServiceRegistry.get_available_workers(self.environment)
        return len(workers)
    
    def _perform_active_recovery(self):
        print(f"[{self.orchestrator_name}] Controllo eventuali job in stato PROCESSING per ripristino...")
        job_ids_to_check = []

        # Usiamo lo state_manager (locale o AWS che sia) come unica fonte di verità
        # per individuare i job orfani, invece di affidarci a pattern di file fisici
        # che non riflettono il reale schema di checkpoint su disco.
        if hasattr(self.state_manager, "get_active_jobs"):
            job_ids_to_check = self.state_manager.get_active_jobs()
        else:
            print(f"[{self.orchestrator_name}] [WARN] Lo state_manager non espone get_active_jobs(): recovery attiva disabilitata.")

        for job_id in job_ids_to_check:
            existing_state = self.state_manager.obtain_request(job_id)
            if existing_state:
                item_data = existing_state.get("Item", existing_state)
                current_status = item_data.get("status")
                old_owner = item_data.get("last_orchestrator")
                if current_status == "PROCESSING" and old_owner != self.orchestrator_name:
                    print(f"[{self.orchestrator_name}] [RECOVERY] Job {job_id[:8]} in stato PROCESSING. Ripristino checkpoint...")
                    print(f"[{self.orchestrator_name}] Il Job {job_id[:8]} era gestito da {old_owner} (mancato).")
                    print(f"[{self.orchestrator_name}] Sincronizzazione stato e subentro immediato in corso...\n")
                    
                    try: 
                        pending_tasks = TaskRegistry.get_tasks_by_job(job_id)
                        by_status = {}
                        for t in pending_tasks:
                            by_status[t.get("status")] = by_status.get(t.get("status"), 0) + 1
                        print(f"[{self.orchestrator_name}] Stato dei task per Job {job_id[:8]}: {by_status}")
                    except Exception as e:
                        print(f"[{self.orchestrator_name}] [WARN] Impossibile recuperare lo stato dei task per Job {job_id[:8]}: {e}")
                    print(f"[{self.orchestrator_name}] Sincronizzazione stato e subentro immediato in corso...\n")
                    job_meta = self._load_job_meta(job_id)
                    recovered_payload = {
                        "job_id": job_id,
                        "request_type": job_meta.get("request_type", "TRAINING"),
                        "dataset_path": job_meta.get("dataset_path") or item_data.get("dataset_path"),
                        "dataset_type": job_meta.get("dataset_type"),
                        "hyperparameters": job_meta.get("hyperparameters", {}),
                        # Vedi _save_job_meta: senza questi, un job federato
                        # ripreso dopo un failover perderebbe la strategia di
                        # partizionamento/allocazione originale.
                        "partition_strategy": job_meta.get("partition_strategy", "iid"),
                        "partition_alpha": job_meta.get("partition_alpha"),
                        "tree_allocation_strategy": job_meta.get("tree_allocation_strategy", "proportional"),
                    }

                    # Il leader precedente è "mancato" (crash), ma la sua lease su
                    # JobLocks resta valida fino alla scadenza naturale del TTL: un
                    # processo morto non può rilasciarla. Se subentrassimo subito con
                    # _process_job, il suo try_claim_job fallirebbe ([CLAIM FAILED] /
                    # [ABORT]) e — chiamato qui con receipt_handle=None — uscirebbe
                    # senza ritentare, lasciando il job orfano fino al timeout esterno.
                    #
                    # Attendiamo quindi attivamente che la lease del vecchio leader
                    # scada, ritentando il claim con un piccolo backoff. Appena il
                    # claim riesce, la lease è nostra: _process_job qui sotto rifarà
                    # try_claim_job, che stavolta la rinnova (refresh_lock) e prosegue
                    # normalmente col recupero dal checkpoint.
                    #
                    # NB: questo NON altera il percorso a regime (non-failover):
                    # riguarda solo il ramo di recovery di un job già PROCESSING
                    # ereditato da un altro orchestrator.
                    lease_wait_timeout = 330   # poco oltre il TTL di 300s della lease su JobLocks
                    lease_poll_interval = 5
                    waited = 0.0
                    claim_ok = False
                    while waited < lease_wait_timeout:
                        if self.state_manager.try_claim_job(job_id, self.orchestrator_name, lease_seconds=300):
                            claim_ok = True
                            break
                        print(f"[{self.orchestrator_name}] [RECOVERY-WAIT] Lease del leader precedente ancora attiva "
                              f"per Job {job_id[:8]}. Nuovo tentativo tra {lease_poll_interval}s "
                              f"(atteso finora: {waited:.0f}s).")
                        time.sleep(lease_poll_interval)
                        waited += lease_poll_interval

                    if not claim_ok:
                        print(f"[{self.orchestrator_name}] [RECOVERY-ABORT] Impossibile acquisire la lease per Job "
                              f"{job_id[:8]} entro {lease_wait_timeout}s: la lascio a un tentativo successivo.")
                        continue

                    try:
                        self._process_job(recovered_payload, receipt_handle=None)
                    except Exception as e:
                        print(f"[{self.orchestrator_name}] Errore durante il recupero del Job {job_id[:8]}: {e}")
    
    # ------------------------------------------------------------------ #
    # Checkpoint INCREMENTALE degli alberi                                #
    # ------------------------------------------------------------------ #
    #
    # FORMATO PRECEDENTE: un'unica chiave, riscritta per intero a ogni chunk
    # con tutta la foresta accumulata fino a quel momento. Rileggerla era
    # banale, ma con W worker le scritture erano W di dimensione crescente
    # (N/W, 2N/W, ... N), per un traffico totale di N*(W+1)/2 invece di N:
    # il costo cresceva col numero di nodi proprio mentre si misurava la
    # scalabilita'.
    #
    # FORMATO ATTUALE: una PARTE per scrittura, contenente solo gli alberi
    # nuovi. Le parti sono numerate in sequenza e non hanno buchi, perche' la
    # scrittura avviene sotto lock e l'indice avanza solo dopo un salvataggio
    # riuscito. Il ripristino puo' quindi enumerarle sondando con exists()
    # finche' non ne trova una mancante: CheckpointDAO espone soltanto
    # save/load/exists/delete, nessuna operazione di listing.
    #
    # RETROCOMPATIBILITA': se non esiste alcuna parte ma esiste il file
    # monolitico del formato precedente, viene letto quello. Alla prima
    # scrittura successiva lo stato viene migrato scrivendo l'intero snapshot
    # come parte 0 e rimuovendo il file monolitico, cosi' i due formati non
    # convivono mai per lo stesso job.
    #
    # Questi metodi usano self.checkpoint_dao e self._resolve_trees_checkpoint_path,
    # definiti nelle sottoclassi (Centralized/Federated): la base fornisce solo
    # la logica, che e' identica per entrambe.

    _TREES_CHECKPOINT_PROBE_MARGIN = 3

    def _trees_checkpoint_part_path(self, job_id: str, index: int) -> str:
        """Path della parte n-esima, derivato da quello del checkpoint monolitico."""
        base = self._resolve_trees_checkpoint_path(job_id)
        stem = base[:-len(".pkl")] if base.endswith(".pkl") else base
        return f"{stem}.part_{index:04d}.pkl"

    def _count_trees_checkpoint_parts(self, job_id: str) -> int:
        """Quante parti risultano gia' persistite per questo job (0 se nessuna)."""
        index = 0
        while self.checkpoint_dao.exists(self._trees_checkpoint_part_path(job_id, index)):
            index += 1
        return index

    def _load_trees_checkpoint(self, job_id: str) -> list:
        """
        Ricompone la foresta parziale leggendo le parti in ordine. Se non ce
        ne sono, ricade sul file monolitico del formato precedente. Ritorna
        una lista vuota se non esiste alcun checkpoint.
        """
        trees, index = [], 0
        while True:
            part_path = self._trees_checkpoint_part_path(job_id, index)
            if not self.checkpoint_dao.exists(part_path):
                break
            trees.extend(self.checkpoint_dao.load(part_path))
            index += 1

        if index > 0:
            print(f"[{self.orchestrator_name}] [CHECKPOINT] Ricomposti {len(trees)} alberi da {index} parti incrementali.")
            return trees

        legacy_path = self._resolve_trees_checkpoint_path(job_id)
        if self.checkpoint_dao.exists(legacy_path):
            legacy_trees = list(self.checkpoint_dao.load(legacy_path))
            print(f"[{self.orchestrator_name}] [CHECKPOINT] Nessuna parte incrementale trovata: letti "
                  f"{len(legacy_trees)} alberi dal checkpoint monolitico (formato precedente).")
            return legacy_trees
        return []

    def _trees_checkpoint_exists(self, job_id: str) -> bool:
        return (self.checkpoint_dao.exists(self._trees_checkpoint_part_path(job_id, 0))
                or self.checkpoint_dao.exists(self._resolve_trees_checkpoint_path(job_id)))

    def _persist_trees_delta(self, job_id: str, snapshot: list, already_persisted: int, part_index: int) -> None:
        """
        Persiste UNA parte. Alla parte 0 scrive l'intero snapshot e rimuove
        l'eventuale file monolitico: e' il passo di migrazione dal formato
        precedente, necessario perche' altrimenti il prefisso gia' presente
        nel monolitico andrebbe perso (il ripristino, trovando delle parti,
        smette di considerare il vecchio file).
        Dalla parte 1 in poi scrive solo il delta non ancora persistito.
        """
        payload = list(snapshot) if part_index == 0 else list(snapshot[already_persisted:])
        self.checkpoint_dao.save(self._trees_checkpoint_part_path(job_id, part_index), payload)
        if part_index == 0:
            legacy_path = self._resolve_trees_checkpoint_path(job_id)
            try:
                if self.checkpoint_dao.exists(legacy_path):
                    self.checkpoint_dao.delete(legacy_path)
            except Exception as e:
                print(f"[{self.orchestrator_name}] [CHECKPOINT WARN] Impossibile rimuovere il checkpoint "
                      f"monolitico dopo la migrazione: {e}")

    def _purge_trees_checkpoint(self, job_id: str) -> None:
        """
        Rimuove tutte le parti e l'eventuale file monolitico. Sonda qualche
        indice oltre la prima assenza (_TREES_CHECKPOINT_PROBE_MARGIN) per non
        lasciare parti orfane di un tentativo precedente: costa qualche
        exists() e evita che un checkpoint incoerente sopravviva alla pulizia.
        """
        index, misses, removed = 0, 0, 0
        while misses < self._TREES_CHECKPOINT_PROBE_MARGIN:
            part_path = self._trees_checkpoint_part_path(job_id, index)
            try:
                if self.checkpoint_dao.exists(part_path):
                    self.checkpoint_dao.delete(part_path)
                    removed += 1
                    misses = 0
                else:
                    misses += 1
            except Exception as e:
                print(f"[{self.orchestrator_name}] [CHECKPOINT WARN] Errore rimuovendo {part_path}: {e}")
                misses += 1
            index += 1
        try:
            self.checkpoint_dao.delete(self._resolve_trees_checkpoint_path(job_id))
        except Exception:
            pass
        if removed:
            print(f"[{self.orchestrator_name}] [CHECKPOINT] Rimosse {removed} parti del checkpoint alberi.")

    def read_selected_features_from_config(self, dataset_type: str = "real"):
        """
        Legge 'feature_selezionate' dal manifesto config_<dataset_type>.json,
        prodotto da run_baseline.py (tuning + feature selection, permutation
        importance OOB + riduzione multicollinearità). Condivisa tra
        centralized.py e FederatedOrchestrator.select_from_config: entrambi i
        percorsi distribuiti devono SOLO applicare questa lista (subset di
        colonne), MAI ricalcolare la feature selection da soli -- rifittare
        un CICIDSFeatureSelector sul lato distribuito duplicherebbe lavoro
        già fatto dalla baseline (un fit di RF + permutation importance),
        e in generale rischierebbe di produrre un set di feature diverso da
        quello della baseline anche a parità di iperparametri, invalidando
        il confronto. Stesso percorso di ricerca file di
        read_decision_threshold_from_config.

        Ritorna None (mai un'eccezione) se il file non esiste o la chiave è
        assente/vuota: il chiamante deve ricadere sul set di feature completo,
        con un avviso esplicito (mai un fallback silenzioso).
        """
        config_filename = f"config_{dataset_type}.json"
        config_path = os.path.join(os.getcwd(), "outputs_baseline", config_filename)

        if not os.path.exists(config_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
            config_path = os.path.join(project_root, "outputs_baseline", config_filename)

        if not os.path.exists(config_path):
            print(f"[{self.orchestrator_name}] [ATTENZIONE] {config_filename} non trovato in "
                  f"nessuno dei percorsi cercati: nessuna feature selezionata da riusare.")
            return None

        try:
            with open(config_path, "r") as f:
                config_dati = json.load(f)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE] Lettura {config_filename} fallita: {e}")
            return None

        feature_selezionate = config_dati.get("feature_selezionate", None)
        if not feature_selezionate:
            print(f"[{self.orchestrator_name}] [ATTENZIONE] 'feature_selezionate' assente o vuoto "
                  f"in {config_filename}.")
            return None

        print(f"[{self.orchestrator_name}] Config caricata da {config_path}. "
              f"Trovate {len(feature_selezionate)} feature.")
        return feature_selezionate

    def read_decision_threshold_from_config(self, dataset_type: str = "real"):
        """
        Legge 'decision_threshold' dal manifesto config_<dataset_type>.json,
        scritto da run_baseline.py DOPO la calibrazione finale della soglia
        (vedi VALIDATION_SIZE_FOR_THRESHOLD in run_baseline.py). Stesso
        pattern/percorso di ricerca file di FederatedOrchestrator.select_from_config
        (che legge 'feature_selezionate' dallo stesso file), qui condiviso in
        BaseOrchestrator perché serve sia al percorso centralizzato sia a
        quello federato.

        Ritorna None (mai un'eccezione) se il file non esiste, non è ancora
        stato aggiornato con la soglia (run di run_baseline.py precedente
        alla sua introduzione), o è malformato: in quel caso il chiamante
        deve ricadere sul comportamento di default (soglia implicita 0.50),
        esattamente come già succede se 'feature_selezionate' è assente.
        """
        config_filename = f"config_{dataset_type}.json"
        config_path = os.path.join(os.getcwd(), "outputs_baseline", config_filename)

        if not os.path.exists(config_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
            config_path = os.path.join(project_root, "outputs_baseline", config_filename)

        if not os.path.exists(config_path):
            print(f"[{self.orchestrator_name}] [ATTENZIONE] {config_filename} non trovato: "
                  f"nessuna soglia di decisione da riusare, ricado sul default (0.50).")
            return None

        try:
            with open(config_path, "r") as f:
                config_dati = json.load(f)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [ERRORE] Lettura {config_filename} fallita: {e}")
            return None

        threshold = config_dati.get("decision_threshold")
        if threshold is None:
            print(f"[{self.orchestrator_name}] [ATTENZIONE] 'decision_threshold' assente in "
                  f"{config_filename} (manifesto prodotto da una versione precedente di "
                  f"run_baseline.py?): ricado sul default (0.50). Rilancia run_baseline.py "
                  f"per rigenerarlo con la soglia inclusa.")
            return None

        print(f"[{self.orchestrator_name}] Soglia di decisione letta da {config_path}: {threshold:.4f}")
        return float(threshold)

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int):
        """
        Implementazione di base: gestisce il checkpoint LOGICO (Metadati su DynamoDB).
        Questo comportamento è comune a TUTTI gli orchestratori.
        """
        print(f"[{self.orchestrator_name}] [BASE-CHECKPOINT] Aggiornamento metadati di stato nel DB.")
        if hasattr(self, 'state_manager') and self.state_manager:
            self.state_manager.update_request_status(
                job_id=job_id, 
                status="PROCESSING", 
                orchestrator_id=self.orchestrator_name, 
                retries=retries,
                base_random_state=base_random_state,
                alberi_addestrati=current_alberi
            )

    def _clean_checkpoint(self, job_id: str):
        if self.environment == "local":
            path = os.path.join("./.local_storage", "checkpoints", f"checkpoint_{job_id}.json")
            if os.path.exists(path):
                os.remove(path)
        self._clean_job_meta(job_id)

    def _save_metrics(self, job_id: str, phase: str, metrics_payload: dict):
        try:
            dao = MetricsDAOFactory.get_dao(self.environment)
            path = self._resolve_metrics_path(job_id, phase)
            dao.save(path, metrics_payload)
            print(f"[{self.orchestrator_name}] [METRICS] Salvate in {path}")
        except Exception as e:
            print(f"[{self.orchestrator_name}] [METRICS-WARN] Salvataggio fallito: {e}")

    def _resolve_metrics_path(self, job_id: str, phase: str) -> str:
        fname = f"{phase}_{job_id}.json"          # phase = "training" | "inference"
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/metrics/{self.__class__.__name__.lower()}/{fname}"
        return os.path.join("./.local_storage/metrics", fname)

    def _aggregate_forest_predictions(
            self,
            predictions_matrix: np.ndarray,
            tree_type: str,
            global_classes: np.ndarray = None,
            decision_threshold: float = None
        ):
            """
            Aggrega le predizioni GREZZE per-albero in un'unica predizione finale per campione.

            Da usare SOLO quando si dispone davvero delle predizioni dei singoli alberi
            (es. modalità centralizzata, dove ogni worker restituisce le predizioni del
            proprio sottoinsieme di alberi). NON va usato quando le predizioni sono già
            state aggregate a monte (es. modalità federata, dove ogni worker restituisce
            direttamente la predizione finale del modello globale sul proprio shard locale):
            in quel caso passare le predizioni finali direttamente a calculate_metrics().

            Classificazione: predictions_matrix ha shape (n_alberi, n_campioni, n_classi_globali)
            e contiene le probabilità per-albero (predict_proba), già allineate allo stesso
            spazio di classi globale. Si fa SOFT VOTING (media delle probabilità), lo stesso
            meccanismo che sklearn usa internamente in RandomForestClassifier.

            decision_threshold: se fornito (solo binario, ignorato altrimenti), la classe
            positiva (etichetta con valore maggiore, es. 1 in 0/1) viene assegnata quando
            la sua probabilità media supera QUESTA soglia, invece del comportamento di
            default (argmax, equivalente a una soglia implicita di 0.5). Serve ad allineare
            l'inferenza distribuita alla soglia calibrata dalla baseline (vedi
            run_baseline.py, VALIDATION_SIZE_FOR_THRESHOLD/decision_threshold in
            config_real.json) invece di ricadere silenziosamente sul default sklearn.
            None (default) -> comportamento invariato rispetto a prima (argmax).

            Regressione: predictions_matrix ha shape (n_alberi, n_campioni) di valori grezzi,
            aggregati con una semplice media (comportamento invariato).
            """
            if tree_type == "classifier":
                if predictions_matrix.ndim != 3:
                    raise ValueError(
                        f"_aggregate_forest_predictions in modalità classificazione richiede una "
                        f"matrice 3D di probabilità per-albero (n_alberi, n_campioni, n_classi), "
                        f"ricevuta shape {predictions_matrix.shape}. Se le predizioni sono già "
                        f"aggregate per campione, chiamare direttamente calculate_metrics()."
                    )
                if global_classes is None:
                    raise ValueError(
                        "_aggregate_forest_predictions in modalità classificazione richiede "
                        "global_classes per tradurre gli indici di colonna nelle etichette reali."
                    )
                global_classes = np.asarray(global_classes)

                # Soft voting: media delle probabilità per-albero sulle stesse colonne di classe.
                avg_proba = np.mean(predictions_matrix, axis=0)

                # y_probs (score continuo per l'AUC) è ben definito solo nel caso binario.
                if len(global_classes) == 2:
                    # Convenzione: l'etichetta con valore maggiore (es. 1 in 0/1) è la classe positiva.
                    positive_idx = int(np.argmax(global_classes))
                    y_probs = avg_proba[:, positive_idx]
                else:
                    y_probs = None

                if decision_threshold is not None and len(global_classes) == 2:
                    negative_idx = 1 - positive_idx
                    positive_label = global_classes[positive_idx]
                    negative_label = global_classes[negative_idx]
                    final_predictions = np.where(
                        y_probs >= decision_threshold, positive_label, negative_label
                    )
                else:
                    # Comportamento di default (invariato): argmax, coerente con
                    # RandomForestClassifier.predict di sklearn -- equivalente a
                    # soglia implicita 0.5 nel caso binario.
                    final_predictions = global_classes[np.argmax(avg_proba, axis=1)]
            else:
                if predictions_matrix.ndim != 2:
                    raise ValueError(
                        f"_aggregate_forest_predictions in modalità regressione richiede una "
                        f"matrice 2D (n_alberi, n_campioni), ricevuta shape {predictions_matrix.shape}. "
                        f"Se le predizioni sono già aggregate per campione, chiamare direttamente "
                        f"calculate_metrics()."
                    )
                final_predictions = np.mean(predictions_matrix, axis=0)
                y_probs = None

            return final_predictions, y_probs

    def _compute_oob_metrics(
            self,
            all_trees: list,
            X_train: np.ndarray,
            y_train: np.ndarray,
            tree_type: str
        ):
            """
            Stima Out-Of-Bag (OOB) dell'errore di generalizzazione (Breiman, 2001).

            Ogni albero bootstrap non vede mai una porzione del training set
            (~36.8% atteso con max_samples=1.0): aggregando le predizioni dei
            SOLI alberi per cui un dato campione era OOB si ottiene una stima
            della performance di generalizzazione "gratis", senza consumare
            il test set separato. Richiede che ogni albero esponga l'attributo
            `oob_sample_indices_` (impostato dai worker in fase di training —
            vedi `_train_single_tree_processor` / `_train_single_fed_tree`).

            Alberi senza questo attributo, o con array OOB vuoto (es. addestrati
            con bootstrap=False), vengono semplicemente esclusi dal calcolo.
            Restituisce None se non c'è materiale sufficiente per una stima
            (nessun albero con indici OOB, o nessun campione mai lasciato fuori).
            """
            n_samples = X_train.shape[0]
            trees_with_oob = [
                t for t in all_trees
                if getattr(t, "oob_sample_indices_", None) is not None and len(t.oob_sample_indices_) > 0
            ]

            if not trees_with_oob:
                print(f"[{self.orchestrator_name}] [OOB] Nessun albero con indici OOB disponibili "
                      f"(bootstrap disattivato o alberi troppo vecchi). Stima OOB saltata.")
                return None

            if tree_type == "classifier":
                trees_with_classes = [t for t in trees_with_oob if hasattr(t, "classes_")]
                if not trees_with_classes:
                    print(f"[{self.orchestrator_name}] [OOB] Nessun albero espone 'classes_'. Stima OOB saltata.")
                    return None
                global_classes = np.unique(np.concatenate([np.asarray(t.classes_) for t in trees_with_classes]))
                n_classes = len(global_classes)

                # Media delle probabilità (soft voting) SOLO fra gli alberi per cui
                # il campione era OOB, coerente con l'aggregazione usata in inferenza.
                proba_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
                oob_count = np.zeros(n_samples, dtype=np.int64)

                for t in trees_with_oob:
                    oob_idx = t.oob_sample_indices_
                    raw_proba = t.predict_proba(X_train[oob_idx])
                    tree_classes = np.asarray(t.classes_)
                    col_positions = np.searchsorted(global_classes, tree_classes)
                    proba_sum[np.ix_(oob_idx, col_positions)] += raw_proba
                    oob_count[oob_idx] += 1

                covered = oob_count > 0
                if not np.any(covered):
                    print(f"[{self.orchestrator_name}] [OOB] Nessun campione ha almeno un albero OOB "
                          f"(troppo pochi alberi?). Stima OOB saltata.")
                    return None

                avg_proba = proba_sum[covered] / oob_count[covered, None]
                oob_predictions = global_classes[np.argmax(avg_proba, axis=1)]
                positive_idx = int(np.argmax(global_classes))
                y_probs = avg_proba[:, positive_idx] if n_classes == 2 else None

                metrics = self.calculate_metrics(
                    final_predictions=oob_predictions,
                    y_test=y_train[covered],
                    tree_type=tree_type,
                    y_probs=y_probs
                )
            else:
                pred_sum = np.zeros(n_samples, dtype=np.float64)
                oob_count = np.zeros(n_samples, dtype=np.int64)

                for t in trees_with_oob:
                    oob_idx = t.oob_sample_indices_
                    pred_sum[oob_idx] += t.predict(X_train[oob_idx])
                    oob_count[oob_idx] += 1

                covered = oob_count > 0
                if not np.any(covered):
                    print(f"[{self.orchestrator_name}] [OOB] Nessun campione ha almeno un albero OOB "
                          f"(troppo pochi alberi?). Stima OOB saltata.")
                    return None

                oob_predictions = pred_sum[covered] / oob_count[covered]
                metrics = self.calculate_metrics(
                    final_predictions=oob_predictions,
                    y_test=y_train[covered],
                    tree_type=tree_type
                )

            coverage = float(np.mean(covered))
            metrics["oob_coverage"] = coverage
            metrics["oob_samples_used"] = int(np.sum(covered))
            metrics["oob_trees_used"] = len(trees_with_oob)
            print(f"[{self.orchestrator_name}] [OOB] Stima calcolata su {int(np.sum(covered))}/{n_samples} "
                  f"campioni ({coverage * 100:.1f}% di copertura) usando {len(trees_with_oob)} alberi.")
            return metrics

    def calculate_metrics(
            self,
            final_predictions: np.ndarray,
            y_test: np.ndarray,
            tree_type: str,
            y_probs: np.ndarray = None
        ):
            """
            Metodo helper per il calcolo, la validazione statistica e la stampa
            delle metriche di performance del modello globale.

            Si aspetta predizioni GIA' finali (una per campione, shape (n_campioni,)):
            - modalità centralizzata: ottenute da _aggregate_forest_predictions()
            - modalità federata: quelle restituite direttamente dai worker
            """
            final_predictions = np.asarray(final_predictions)
            y_test = np.asarray(y_test)
            if final_predictions.shape != y_test.shape:
                raise ValueError(
                    f"final_predictions {final_predictions.shape} e y_test {y_test.shape} devono avere "
                    f"la stessa shape (una predizione per campione). Se si sta passando una matrice "
                    f"grezza per-albero, aggregarla prima con _aggregate_forest_predictions()."
                )

            if tree_type == "classifier":
                final_predictions = final_predictions.astype(int)
                y_test = y_test.astype(int)

                n_classes = len(np.unique(np.concatenate([y_test, final_predictions])))
                avg_method = "binary" if n_classes <= 2 else "weighted"

                # Calcolo delle metriche di classificazione standard
                accuracy = np.mean(final_predictions == y_test)
                precision = precision_score(y_test, final_predictions, average=avg_method, zero_division=0)
                recall = recall_score(y_test, final_predictions, average=avg_method, zero_division=0)
                f1 = f1_score(y_test, final_predictions, average=avg_method, zero_division=0)
                # L'AUC richiede degli score/probabilità continui: se il chiamante non ne dispone
                # (es. federato, se i worker non restituiscono anche predict_proba) resta None.
                auc = roc_auc_score(y_test, y_probs) if (n_classes == 2 and y_probs is not None) else None
                cm = confusion_matrix(y_test, final_predictions)


                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc": auc,
                    "confusion_matrix": cm.tolist(),
                    "classification_report": classification_report(y_test, final_predictions, output_dict=True, zero_division=0)
                }

            else:
                mse = mean_squared_error(y_test, final_predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, final_predictions)
                r2 = r2_score(y_test, final_predictions)

                metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }

            return metrics