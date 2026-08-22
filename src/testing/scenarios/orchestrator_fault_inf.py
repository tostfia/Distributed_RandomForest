import time
import threading
import os
import re
import json
from src.testing.scenarios.base import BaseTestScenario
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator
import docker

# ---------------------------------------------------------------------
# AWS: nomi/tabelle usati per il failover REALE sui 2 task ECS del
# orchestrator-service già dispiegato (vedi _run_aws_real_failover).
# ---------------------------------------------------------------------
_LOCK_TABLE = "OrchestratorLocks"
_LOCK_KEY = "global_orchestrator_leader_lock"


def _merge_aws_overrides(config: dict, key: str) -> dict:
    """
    Unisce il blocco di config 'key' (es. 'inference_orchestrator_failover')
    con l'eventuale override AWS-specifico in
    config['aws']['suggested_overrides'][key] — quest'ultimo, se presente,
    vince sui valori "locali". Su AWS il ciclo ETL/RPC reale e il tempo di
    ecs.stop_task() sono più lenti del kill istantaneo locale/Docker, quindi
    i timeout tarati per il locale possono essere troppo stretti. Filtra le
    chiavi di solo commento (es. '_NOTE') presenti nel JSON di config.
    """
    merged = dict(config.get(key, {}) or {})
    if (config.get("aws", {}) or {}).get("suggested_overrides", {}).get(key):
        overrides = config["aws"]["suggested_overrides"][key]
        merged.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return merged


def _resolve_aws_infra(config: dict):
    """
    Cluster/region/nome service ECS: letti da config['aws'] quando presente
    (stessa sezione già usata da run_test_aws.sh/aws_ecs_utils.py, vedi
    test_config.json), altrimenti fallback su env var/default fisso.
    """
    aws_cfg = config.get("aws", {}) or {}
    cluster = aws_cfg.get("ecs_cluster_name") or os.environ.get("CLUSTER_NAME", "forest-cluster")
    region = aws_cfg.get("region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    orch_service = aws_cfg.get("orchestrator_service_name", "orchestrator-service")
    return cluster, region, orch_service


def _wait_for_leadership(orch, timeout=15, interval=0.5) -> bool:
    """
    Attende (con timeout) che 'orch' risulti effettivamente leader, prima di
    procedere: senza questo controllo il test può inviare il job di inferenza
    e simulare il crash mentre 'orch' non ha ancora davvero acquisito la
    leadership (o non l'acquisisce affatto), rendendo il "kill del leader"
    un'operazione su un processo che di fatto non stava facendo nulla.

    Usata SOLO dai rami locale/Docker Compose. Il ramo AWS (vedi
    _run_aws_real_failover) non la usa: lì Leader e Standby sono già due
    task ECS reali, sempre attivi, indipendenti dal test-engine.
    """
    lock_key = orch._get_lock_key()
    waited = 0.0
    while waited < timeout:
        if orch.environment == "local":
            lock_path = os.path.join("./.local_storage", f"{lock_key}.json")
            if os.path.exists(lock_path):
                try:
                    with open(lock_path, "r", encoding="utf-8") as f:
                        lock_data = json.load(f)
                    if lock_data.get("leader") == orch.orchestrator_name:
                        return True
                except (json.JSONDecodeError, KeyError, OSError):
                    pass
        else:
            try:
                if orch._try_acquire_leadership():
                    return True
            except Exception:
                pass
        time.sleep(interval)
        waited += interval
    return False

def _count_checkpoint_workers(checkpoint_path: str) -> int:
    """
    Conta quanti worker/chunk sono già salvati nel checkpoint di inferenza,
    leggendolo direttamente dal file pickle (la LocalCheckpointDAO serializza
    con pickle puro). Ritorna 0 se il file non esiste, non è ancora leggibile,
    o è in corso di scrittura (la scrittura atomica tmp+os.replace rende questa
    finestra minima, ma un tentativo può comunque fallire: in tal caso 0 e si
    riprova al giro dopo).

    Gestisce entrambe le strutture possibili del checkpoint:
      - federato: dict {worker_index: {...}}  -> len(dict)
      - centralizzato: list di chunk          -> len(list)

    Usata SOLO dai rami locale/Docker Compose (checkpoint su bind mount
    condiviso). Il ramo AWS usa _count_completed_worker_tasks_aws, che legge
    lo stesso tipo di segnale da DynamoDB (tabella WorkerTasks) invece che
    da un file locale — su Fargate non esiste alcun filesystem condiviso
    tra il test-engine e i task reali di orchestrator-service.
    """
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        return 0
    try:
        import pickle
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return len(data)
        if isinstance(data, list):
            return len(data)
        return 0
    except Exception:
        # File in scrittura, troncato o non ancora valido: riprova al giro dopo.
        return 0


def _wait_for_inference_in_progress(job_id, timeout=60, interval=0.2,
                                    min_workers=1, expected_workers=None,
                                    require_partial=False) -> float:
    """
    Attende (con timeout) che l'inferenza distribuita sia in corso e, se
    richiesto, che sia in una FINESTRA DI RIPRESA PARZIALE: almeno `min_workers`
    worker già validati (c'è lavoro salvato da riprendere) ma non ancora tutti
    (`expected_workers`), così che uccidere il leader ORA lasci allo standby dei
    risultati da riprendere via checkpoint E dei worker ancora da completare.

    Perché serve la finestra parziale (require_partial=True): nel federato
    l'inferenza è velocissima. Se ci si limita ad attendere "l'inferenza è
    partita" (marcatore) e si uccide subito, il leader muore prima che QUALSIASI
    worker abbia salvato: lo standby non trova nulla da riprendere e rifà tutto
    da zero — un failover valido, ma che NON esercita lo SHORT-CIRCUIT della
    ripresa. Aspettando invece che il checkpoint contenga tra 1 e N-1 worker, il
    crash cade nel punto in cui la ripresa parziale è realmente dimostrabile.

    Segnali osservati (ambiente 'local', via bind mount):
      - checkpoint 'inference_chunks_{job_id}.pkl' (federato: dict per-worker;
        centralizzato: lista di chunk) — contato per sapere quanti worker sono
        pronti;
      - marcatore 'inference_started_{job_id}.marker' — solo come segnale di
        "inferenza avviata", usato quando require_partial=False.

    Usata SOLO dai rami locale/Docker Compose. Il ramo AWS usa l'analogo
    _wait_for_inference_in_progress_aws, basato su query DynamoDB (tabella
    WorkerTasks, GSI job_id-index) invece che su file locali.

    Ritorna i secondi attesi al raggiungimento della condizione, o -1.0 se scade
    il timeout. Con require_partial=True il timeout include anche il caso in cui
    l'inferenza è passata da 0 a "tutti i worker" senza mai essere osservata in
    stato parziale (finestra troppo stretta): in tal caso il chiamante applica
    il proprio fallback.
    """
    checkpoint_path = f"./.local_storage/inference_chunks_{job_id}.pkl"
    marker_path = f"./.local_storage/inference_started_{job_id}.marker"
    waited = 0.0
    while waited < timeout:
        n_done = _count_checkpoint_workers(checkpoint_path)

        if require_partial:
            # Finestra parziale: almeno min_workers salvati, ma non tutti.
            upper_ok = (expected_workers is None) or (n_done < expected_workers)
            if n_done >= min_workers and upper_ok:
                return waited
        else:
            # Modalità "inferenza avviata": basta un segnale qualsiasi.
            if n_done >= min_workers or os.path.exists(marker_path):
                return waited

        time.sleep(interval)
        waited += interval
    return -1.0


def _count_completed_worker_tasks_aws(state_manager, job_id: str) -> int:
    """
    Equivalente AWS di _count_checkpoint_workers: conta quanti task worker
    risultano COMPLETED per questo job_id, leggendo la tabella DynamoDB
    WorkerTasks tramite il suo GSI 'job_id-index' (stesso meccanismo già
    usato in produzione da AwsStateManager.are_all_workers_done). Sia
    centralized.py che federated.py chiamano _track_task(status="COMPLETED")
    per ogni worker anche durante l'inferenza (non solo il training), quindi
    il segnale è valido per questo scenario.
    """
    try:
        response = state_manager._db.query_by_index(
            table_name="WorkerTasks", index_name="job_id-index",
            key_name="job_id", key_value=job_id
        )
        items = response.get("Items", [])
        return len([t for t in items if t.get("status") == "COMPLETED"])
    except Exception:
        return 0


def _wait_for_inference_in_progress_aws(state_manager, job_id, timeout=60, interval=1.0,
                                         min_workers=1, expected_workers=None,
                                         require_partial=False) -> float:
    """
    Equivalente AWS di _wait_for_inference_in_progress: stessa logica di
    attesa di una finestra di ripresa parziale, ma il conteggio dei worker
    già completati arriva da DynamoDB (WorkerTasks) invece che da un
    checkpoint pickle locale, perché su Fargate non c'è alcun filesystem
    condiviso tra il test-engine e i task reali del orchestrator-service.
    """
    waited = 0.0
    while waited < timeout:
        n_done = _count_completed_worker_tasks_aws(state_manager, job_id)

        if require_partial:
            upper_ok = (expected_workers is None) or (n_done < expected_workers)
            if n_done >= min_workers and upper_ok:
                return waited
        else:
            if n_done >= min_workers:
                return waited

        time.sleep(interval)
        waited += interval
    return -1.0


def _resolve_leader_container(client, lock_dir="./.local_storage"):
    """
    Determina QUALE dei container Docker (tra 'orchestrator-1' e
    'orchestrator-2') sta effettivamente detenendo la leadership in questo
    momento, leggendo il lock condiviso su disco — invece di assumere
    staticamente che sia sempre 'orchestrator-1'.

    L'elezione della leadership tra i due container è non deterministica:
    dipende da chi acquisisce per primo il lock all'avvio (vedi
    _try_acquire_leadership in BaseOrchestrator). Uccidere sempre lo stesso
    nome per posizione rischia quindi di colpire lo STANDBY invece del
    LEADER, vanificando il test: la richiesta di inferenza continuerebbe
    indisturbata sul leader originale e il test riporterebbe SUCCESS senza
    aver testato alcun failover reale.

    Il campo 'leader' nel lock contiene il nome interno dell'orchestratore
    (es. 'Orchestrator-local-centralized-7a4fd6802b7b-1'), che include
    l'hostname del container — che Docker imposta di default all'ID breve
    del container (12 caratteri esadecimali). Estraiamo quell'ID e lo
    confrontiamo con l'ID reale dei container in esecuzione per capire
    quale dei due sia il vero leader.

    Ritorna l'oggetto container Docker del leader, o None se non è stato
    possibile determinarlo (lock assente/corrotto, o nessun container
    corrispondente trovato).
    """
    lock_key = "global_orchestrator_leader_lock"
    lock_path = os.path.join(lock_dir, f"{lock_key}.json")
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            lock_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    leader_name = lock_data.get("leader", "")
    match = re.search(r"[0-9a-f]{12}", leader_name.lower())
    if not match:
        return None
    hostname_fragment = match.group(0)

    containers = client.containers.list(filters={
        "label": "com.docker.compose.service=orchestrator"
    })
    for c in containers:
        if c.id.startswith(hostname_fragment):
            return c
    return None


def _get_current_leader_name_aws(state_manager):
    """
    Legge il nome dell'orchestratore che detiene ATTUALMENTE il lock di
    leadership su DynamoDB (tabella OrchestratorLocks, chiave 'lock_key' =
    'global_orchestrator_leader_lock', campo 'leader' — vedi
    AwsDynamoDB.try_acquire_lock in dynamodb_aws.py per lo schema esatto).
    Ritorna None se il lock non esiste o la lettura fallisce.
    """
    try:
        item = state_manager._db.get_item(_LOCK_TABLE, _LOCK_KEY)
        return item.get("Item", {}).get("leader")
    except Exception as e:
        print(f"[TEST ERRORE] Lettura del lock di leadership da DynamoDB fallita: {e}")
        return None


def _resolve_ecs_task_arn_by_ip(ecs_client, cluster, service_name, ip_dashed):
    """
    Trova, tra i task RUNNING del service ECS indicato, quello il cui IP
    privato (formato Fargate awsvpc, es. '172.31.70.125') corrisponde al
    frammento '172-31-70-125' estratto dal nome interno dell'orchestratore
    (che include l'hostname Fargate, identico al pattern già usato per i
    worker). Ritorna l'ARN del task, o None se non trovato.

    NOTA 'ip_dashed': deve essere SENZA il prefisso 'ip-' (es. '172-31-70-125'),
    perché va confrontato con 'ip.replace(".", "-")' qui sotto, che produce un
    IP puro senza prefisso. Passare un valore con il prefisso 'ip-' fa fallire
    SEMPRE il confronto, anche quando il task cercato esiste davvero.
    """
    task_arns = ecs_client.list_tasks(
        cluster=cluster, serviceName=service_name, desiredStatus="RUNNING"
    ).get("taskArns", [])
    if not task_arns:
        return None

    details = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns).get("tasks", [])
    for t in details:
        for att in t.get("attachments", []):
            for d in att.get("details", []):
                if d.get("name") == "privateIPv4Address":
                    ip = d.get("value", "")
                    if ip and ip.replace(".", "-") == ip_dashed:
                        return t.get("taskArn")
    return None


class InferenceOrchestratorFaultScenario(BaseTestScenario):
    """
    Copre lo Scenario: Failover dell'Orchestratore durante la fase di inferenza.

    Su locale/Docker Compose: usa l'orchestratore nativo del TestEngine come
    Leader (Master-1) e istanzia un secondo orchestratore di Standby
    (Master-2) per il subentro — invariato rispetto alla versione originale.

    Su AWS: NON istanzia nulla in-process. Usa il vero orchestrator-service
    già dispiegato (2 task ECS, Leader+Standby sempre attivi) — vedi
    _run_aws_real_failover per il dettaglio.
    """

    def run(self) -> dict:

        ft_cfg = _merge_aws_overrides(self.config, "inference_orchestrator_failover")
        # Numero di alberi dal manifesto della baseline (vedi
        # BaseTestScenario._resolve_hyperparameters): stessa fonte del payload,
        # quindi non si puo' piu' chiedere N alberi dichiarandone M ai worker.
        target_trees = self._resolve_target_trees()

        orch_leader = self.orchestrator
        environment = getattr(orch_leader, "environment", "local")

        # AWS: percorso completamente separato, vedi _run_aws_real_failover.
        if environment == "aws":
            return self._run_aws_real_failover(orch_leader, ft_cfg, target_trees)

        print(f"\n--- [TEST] Failover dell'Orchestratore durante l'inferenza ' ---")
        orchestrator_type = os.environ.get("SYS_MODE", "centralized")
        target_queue = orch_leader.queue_name
        docker_env = os.environ.get("RUNNING_IN_DOCKER")
        if docker_env == "true":
            print("[TEST] Ambiente DOCKER rilevato: avviato già avviato il secondo orchestratore come container...")
            if orchestrator_type == "federated":
                orch_standby = FederatedOrchestrator(orchestrator_name="distributed_randomforest-orchestrator-2")
            else:
                orch_standby = CentralizedOrchestrator(orchestrator_name="distributed_randomforest-orchestrator-2")
            orch_standby.queue_name = target_queue
        else:
            if orchestrator_type == "federated":
                print("[TEST] Istanzio l'Orchestratore FEDERATO di Standby ...")
                orch_standby = FederatedOrchestrator(orchestrator_name="distributed_randomforest-orchestrator-2")
                orch_standby.queue_name = target_queue
            else:
                print("[TEST] Istanzio l'Orchestratore CENTRALIZZATI di Standby ...")
                orch_standby = CentralizedOrchestrator(orchestrator_name="distributed_randomforest-orchestrator-2")
                orch_standby.queue_name = target_queue

            # In locale (non-Docker) 'orch_leader' non è un processo esterno già
            # attivo: va avviato qui, PRIMA dello standby, e va atteso che diventi
            # davvero leader — altrimenti il crash simulato più sotto colpirebbe
            # un orchestratore che di fatto non stava servendo la richiesta di
            # inferenza (l'avrebbe presa in carico solo lo standby).
            leader_thread = threading.Thread(target=orch_leader.start, name="LeaderThread")
            leader_thread.daemon = True
            leader_thread.start()

            if not _wait_for_leadership(orch_leader, timeout=15):
                print(f"[TEST ERRORE] '{orch_leader.orchestrator_name}' non ha acquisito la leadership entro il timeout: "
                      f"il test non può simulare un failover credibile.")
                return {"status": "FAILED", "duration_seconds": 0,
                        "error": "Il leader designato non ha acquisito la leadership entro il timeout."}
            print(f"[TEST] '{orch_leader.orchestrator_name}' ha acquisito la leadership. Avvio dello Standby...")

            standby_thread = threading.Thread(target=orch_standby.start, name="StandbyThread")
            standby_thread.daemon = True
            standby_thread.start()

            time.sleep(2)  # Diamo tempo allo standby di assestarsi

        job_id = f"test_inference_orch_failover_{int(time.time())}"
        # Iperparametri dal manifesto della baseline: fonte unica condivisa
        # con run_baseline() (vedi BaseTestScenario._resolve_hyperparameters).
        hp = self._resolve_hyperparameters()
        payload = {
            "job_id": job_id,
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": hp,
        }
        if orchestrator_type == "federated":
            payload = self._augment_payload_with_partitioning(payload)

        try:
            self._reuse_dataset_if_available(payload, seed=123)
            orch_leader._execute_training_step(payload, 0, target_trees, 123)

        except Exception as e:
            print(f"[TEST ERRORE] Setup fallito: {e}")
            return {"status": "FAILED", "duration_seconds": 0}

        payload_inference = {
            "request_type": "INFERENCE",
            "job_id": job_id,
            "data_path": self.config["dataset_path"],
            "dataset_type": self.config["dataset_type"],
            "hyperparameters": hp,
        }
        if orchestrator_type == "federated":
            payload_inference = self._augment_payload_with_partitioning(payload_inference)
        print(f"[TEST] Invio del Job {job_id[:8]} alla coda '{orch_leader.queue_name}'...")

        try:
            orch_leader.sqs_queue.send_message(queue_name=orch_leader.queue_name, message_dict=payload_inference)
        except Exception as e_inner:
            print(f"[TEST ERRORE CRITICO] Impossibile inviare il messaggio: {e_inner}")
            return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0}
        start_time = time.perf_counter()

        def _simulate_backend_unreachable(orch):
            """
            Simula la perdita di accesso al coordination backend (DynamoDB/mock)
            da parte di 'orch': la sua prossima chiamata a try_claim_job() farà
            scattare l'abort già previsto dal codice (MessageOwnershipLostError),
            invece di lasciarlo lavorare indisturbato in background nonostante
            il "crash" simulato.
            """
            def _denied(*args, **kwargs):
                return False
            orch.state_manager.try_claim_job = _denied

        def kill_orchestrator():
            kill_delay = ft_cfg.get("kill_orchestrator_after_seconds", 5)
            signaled = self.orchestrator.chunk_sent_event.wait(timeout=kill_delay)
            if not signaled:
                print(f"[TEST WARN] Timeout di {kill_delay} secondi raggiunto senza che il chunk sia stato inviato. Procedo comunque a simulare il guasto.")
            print(f"[TEST KILLER] Lascio lavorare il leader per {kill_delay} secondi prima del crash...")

            print(f"\n[TEST TRIGGER] !!! SIMULAZIONE CRASH IMPREVISTO DI {orch_leader.orchestrator_name.upper()} !!!")

            # Blocchiamo l'heartbeat PRIMA di toccare il lock: se lo rimuovessimo
            # con l'heartbeat ancora vivo, quest'ultimo lo ricreerebbe da solo
            # entro pochi secondi (_refresh_leadership_lock non distingue "l'ho
            # perso io" da "il file manca, lo riscrivo").
            orch_leader._stop_heartbeat.set()

            # Il lavoro già in corso smette di essere "silenzioso": al prossimo
            # controllo di lease lo rileverà e abortirà da solo.
            _simulate_backend_unreachable(orch_leader)

            # Il patch sopra impedisce solo AL LEADER di riconquistare la lease,
            # ma il lock reale su JobLocks resta valido fino al suo TTL naturale
            # (300s): senza rilasciarlo esplicitamente qui, lo standby otterrebbe
            # sempre CLAIM FAILED finché quel TTL non scade da solo.
            try:
                orch_leader.state_manager.release_job_lease(job_id, orch_leader.orchestrator_name)
                print(f"[TEST TRIGGER] Job lease di '{job_id[:8]}' rilasciata forzatamente.")
            except Exception:
                pass

            lock_key = orch_leader._get_lock_key()
            if orch_leader.environment == "local":
                lock_path = os.path.join("./.local_storage", f"{lock_key}.json")
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                        print(f"[TEST TRIGGER] Lock '{lock_key}' rimosso dal File System.")
                    except Exception:
                        pass
            else:
                try:
                    orch_leader.state_manager.release_global_lock(lock_key, orch_leader.orchestrator_name)
                    print(f"[TEST TRIGGER] Lock '{lock_key}' rilasciato da DynamoDB.")
                except Exception:
                    pass

            print(f"[TEST TRIGGER] '{orch_leader.orchestrator_name}' è stato neutralizzato.")

        # Contenitore mutabile condiviso col killer thread: registra l'ID (breve)
        # del container leader effettivamente ucciso, così la verifica finale può
        # controllare che a completare il job sia stato un ALTRO orchestratore
        # (il vero standby subentrato), non il leader originale — che sarebbe un
        # falso positivo (kill fuori finestra, job chiuso dallo stesso leader).
        killed_leader_info = {"container_id": None}

        def kill_docker_leader_target():
            # Vogliamo che il crash cada nella FINESTRA DI RIPRESA PARZIALE:
            # almeno un worker già validato e salvato nel checkpoint (c'è lavoro
            # da riprendere) ma non ancora tutti (resta lavoro da completare). È
            # l'unico punto in cui lo standby esercita davvero lo SHORT-CIRCUIT
            # della ripresa, invece di rifare l'inferenza da zero.
            wait_timeout = ft_cfg.get("max_wait_for_inference_start_seconds", 60)
            expected_workers = getattr(orch_leader, "num_workers", None)
            # Polling stretto: nel federato l'inferenza è velocissima, la finestra
            # 1..N-1 dura poco, quindi campioniamo spesso.
            poll_interval = ft_cfg.get("inference_poll_interval_seconds", 0.05)

            waited = _wait_for_inference_in_progress(
                job_id, timeout=wait_timeout, interval=poll_interval,
                min_workers=1, expected_workers=expected_workers, require_partial=True
            )

            if waited >= 0:
                print(f"[TEST KILLER] Finestra di ripresa parziale rilevata dopo {waited:.2f}s "
                      f"(>=1 worker nel checkpoint, non ancora tutti). Simulo il crash: "
                      f"lo standby dovrà riprendere il lavoro già salvato e completare il resto.")
            else:
                # Fallback 1: la finestra parziale non è stata colta (inferenza
                # troppo rapida, passata da 0 a N senza stato intermedio
                # osservabile). Ripieghiamo su "inferenza avviata": almeno il
                # failover viene esercitato, anche se lo standby potrebbe rifare
                # tutto da zero invece di riprendere parzialmente.
                waited = _wait_for_inference_in_progress(
                    job_id, timeout=wait_timeout, interval=poll_interval,
                    min_workers=1, require_partial=False
                )
                if waited >= 0:
                    print(f"[TEST KILLER] [WARN] Finestra parziale non colta (inferenza troppo rapida): "
                          f"il crash cade su 'inferenza avviata' dopo {waited:.2f}s. Il failover è comunque "
                          f"testato, ma la ripresa PARZIALE da checkpoint potrebbe non essere esercitata.")
                else:
                    # Fallback 2: nessun segnale entro il timeout. Ultimo ripiego:
                    # ritardo fisso, con avviso che il crash potrebbe cadere fuori
                    # dalla finestra utile.
                    kill_delay = ft_cfg.get("docker_kill_delay_seconds", 1)
                    print(f"[TEST KILLER] [WARN] Nessun segnale di inferenza entro {wait_timeout}s: "
                          f"ripiego su un ritardo fisso di {kill_delay}s. Verificare il risultato.")
                    time.sleep(kill_delay)

            print(f"\n[TEST TRIGGER] !!! SIMULAZIONE CRASH IMPREVISTO (DOCKER) !!!")

            client = docker.from_env()
            target_container = _resolve_leader_container(client)

            if target_container is None:
                # Fallback: lock non leggibile (assente/corrotto). Proviamo
                # comunque 'orchestrator-1', ma segnaliamo l'incertezza — meglio
                # un test che dichiara la propria ambiguità che uno che riporta
                # SUCCESS avendo ucciso lo standby per sbaglio.
                print("[TEST TRIGGER WARN] Impossibile identificare il leader dal lock "
                      "condiviso. Fallback su 'orchestrator-1' (potrebbe essere lo "
                      "standby, se ha vinto l'elezione 'orchestrator-2'): il risultato "
                      "di questo test va verificato manualmente.")
                containers = client.containers.list(filters={
                    "label": "com.docker.compose.service=orchestrator"
                })
                for c in containers:
                    if "orchestrator-1" in c.name:
                        target_container = c
                        break

            found = target_container is not None
            if found:
                killed_leader_info["container_id"] = target_container.id
                print(f"[TEST TRIGGER] Leader identificato dal lock: {target_container.name}. "
                      f"Disattivo la restart policy per evitare che risorga da solo...")
                target_container.update(restart_policy={"Name": "no"})
                print(f"[TEST TRIGGER] Kill fisico del container: {target_container.name}")
                target_container.kill()
            else:
                print("[TEST TRIGGER ERROR] Nessun container leader trovato tra i container attivi.")
            # Container morto per davvero: qui non c'è un heartbeat in-process da
            # fermare prima, resta solo da ripulire il lock che il container
            # ucciso non ha potuto rilasciare da sé.
            lock_key = orch_leader._get_lock_key()
            if orch_leader.environment == "local":
                lock_path = os.path.join("./.local_storage", f"{lock_key}.json")
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                        print(f"[TEST TRIGGER] Lock '{lock_key}' rimosso dal File System.")
                    except Exception:
                        pass
            else:
                try:
                    orch_leader.state_manager.release_global_lock(lock_key, orch_leader.orchestrator_name)
                    print(f"[TEST TRIGGER] Lock '{lock_key}' rilasciato da DynamoDB.")
                except Exception:
                    pass
            print(f"[TEST TRIGGER] '{orch_leader.orchestrator_name}' [Docker] è stato neutralizzato.")

        if docker_env != "true":
            killer_thread = threading.Thread(target=kill_orchestrator, name="KillerThread")
            killer_thread.daemon = True
            killer_thread.start()
        else:
            print(f"[TEST] Ambiente DOCKER rilevato: il crash del leader sarà simulato dal container .")
            killer_thread = threading.Thread(target=kill_docker_leader_target, name="DockerKillerThread")
            killer_thread.daemon = True
            killer_thread.start()

        # Il kill ora cade quando l'inferenza è genuinamente in corso (checkpoint
        # con almeno un chunk presente). Come nel failover di training, lo standby
        # che subentra potrebbe dover attendere la scadenza naturale della job-lease
        # del leader morto (fino a ~300s) prima di poter reclamare il job — vedi il
        # retry [RECOVERY-WAIT] in BaseOrchestrator._perform_active_recovery. Il
        # timeout di monitoraggio deve quindi lasciare margine per quell'attesa più
        # il recovery vero e proprio (rapido, perché i chunk già fatti vengono
        # ripresi dal checkpoint via SHORT-CIRCUIT).
        failover_detection_margin = ft_cfg.get("failover_detection_margin_seconds", 90)
        max_timeout = ft_cfg.get("max_monitor_timeout_seconds", failover_detection_margin + 360)
        job_completed = False
        completed_by = None   # orchestrator_id che ha portato il job a COMPLETED
        trees_built = 0
        max_trees_built_seen = 0  # tiene il massimo, perché al COMPLETED il campo può azzerarsi
        elapsed = 0
        check_interval = 3

        print(f"[TEST] Monitoraggio dello stato del Job (timeout: {max_timeout}s)...")
        while elapsed < max_timeout:
            time.sleep(check_interval)
            elapsed += check_interval

            try:
                state = orch_standby.state_manager.obtain_request(job_id)
                if state:
                    item_data = state.get("Item", state)
                    status = item_data.get("status")
                    trees_built = item_data.get("alberi_addestrati", 0)
                    max_trees_built_seen = max(max_trees_built_seen, trees_built)
                    current_owner = item_data.get("orchestrator_id") or item_data.get("last_orchestrator")

                    print(f"[TEST MONITOR] Stato: {status} | Alberi a checkpoint: {trees_built} | Gestore Attuale: {current_owner}")

                    if status == "COMPLETED":
                        job_completed = True
                        completed_by = current_owner
                        break
            except Exception as e:
                print(f"[TEST MONITOR WARNING] Errore lettura stato: {e}")

        # Verifica che il completamento sia opera dello STANDBY subentrato, non del
        # leader originale ucciso. Il nome interno dell'orchestratore include
        # l'hostname del container (= il suo ID breve), quindi se l'ID del container
        # killato compare nel nome del completatore, il job è stato chiuso dallo
        # stesso leader → il kill è caduto fuori finestra e il failover NON è stato
        # realmente esercitato (falso positivo da respingere).
        killed_id = killed_leader_info.get("container_id")
        completed_by_killed_leader = bool(
            killed_id and completed_by and killed_id[:12] in completed_by.lower()
        )
        if completed_by_killed_leader:
            print(f"[TEST WARN] Il job risulta completato dallo stesso leader ucciso ({completed_by}): "
                  f"il crash è probabilmente caduto a inferenza già conclusa. Failover NON esercitato.")

        # Il modello viene sempre salvato come "model_{job_id}.pkl" in entrambe le
        # modalità (vedi _resolve_model_path in centralized.py e federated.py).
        path_modello_dinamico = f"./saved_models/model_{job_id}.pkl"

        if job_completed and not completed_by_killed_leader:
            test_status = "SUCCESS"
            print(f"\n[TEST PASSED] Failover completato! Job chiuso dallo standby subentrato ({completed_by}).")
        else:
            test_status = "FAILED"
            print(f"\n[TEST FAILED] Failover non verificato "
                  f"({'completato dal leader ucciso' if completed_by_killed_leader else 'job non arrivato a COMPLETED entro il timeout'}).")
        duration = time.perf_counter() - start_time

        return {
            "scenario_description": "Test di Failover dell'Orchestratore con transizione della leadership dello stato su crash del Master.",
            "status": test_status,
            "original_leader": orch_leader.orchestrator_name,
            "recovered_by": completed_by if (job_completed and not completed_by_killed_leader) else "None",
            "duration_seconds": duration
        }

    # ------------------------------------------------------------------ #
    # AWS: failover REALE sui 2 task ECS del orchestrator-service         #
    # ------------------------------------------------------------------ #

    def _run_aws_real_failover(self, orch_leader, ft_cfg, target_trees) -> dict:
        """
        Failover REALE durante l'inferenza sui due task ECS del
        orchestrator-service già dispiegato. Setup (generazione del modello
        preliminare) fatto in-process via chiamata diretta, ESATTAMENTE come
        fanno già fault_inf.py/network.py — non richiede failover, serve solo
        a produrre il .pkl su S3. La richiesta di INFERENZA vera e propria
        invece passa da SQS, così la reclama il leader reale su ECS.

        Per la finestra di ripresa parziale, il segnale non può più venire da
        un checkpoint locale (nessun filesystem condiviso col test-engine su
        Fargate): usiamo invece la tabella DynamoDB WorkerTasks (GSI
        job_id-index), popolata da _track_task() anche durante l'inferenza
        (vedi centralized.py/federated.py) — stesso principio, fonte diversa.

        Nessuna scorciatoia su lock/lease: il recovery si affida al TTL
        naturale, come nella versione training (vedi _run_aws_real_failover
        in orchestrator_fault.py per il dettaglio del meccanismo).
        """
        import boto3

        cluster, region, service_name = _resolve_aws_infra(self.config)

        print(f"\n--- [TEST] Failover REALE dell'Orchestratore durante l'inferenza su ECS "
              f"('{service_name}', cluster '{cluster}') ---")

        job_id = f"test_inference_orch_failover_aws_{int(time.time())}"
        # Iperparametri dal manifesto della baseline: fonte unica condivisa
        # con run_baseline() (vedi BaseTestScenario._resolve_hyperparameters).
        hp = self._resolve_hyperparameters()
        payload = {
            "job_id": job_id,
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": hp,
        }
        # isinstance su orch_leader (non os.environ): questo metodo, come in
        # orchestrator_fault.py, non ha garanzia che SYS_MODE rifletta lo
        # stesso valore usato per istanziare l'orchestratore ricevuto.
        is_federated = isinstance(orch_leader, FederatedOrchestrator)
        if is_federated:
            payload = self._augment_payload_with_partitioning(payload)

        print(f"\n[TEST] Setup: addestramento preliminare in-process (nessun failover in questa fase, "
              f"serve solo a produrre il modello su S3)...")
        try:
            self._reuse_dataset_if_available(payload, seed=123)
            orch_leader._execute_training_step(payload, 0, target_trees, 123)
        except Exception as e:
            print(f"[TEST ERRORE] Setup fallito: {e}")
            return {"status": "FAILED", "duration_seconds": 0}

        payload_inference = {
            "request_type": "INFERENCE",
            "job_id": job_id,
            "data_path": self.config["dataset_path"],
            "dataset_type": self.config["dataset_type"],
            "hyperparameters": hp,
        }
        if is_federated:
            payload_inference = self._augment_payload_with_partitioning(payload_inference)
        print(f"[TEST] Invio della richiesta di INFERENZA per il Job {job_id[:8]} alla coda "
              f"'{orch_leader.queue_name}' (la reclamerà il leader reale su ECS)...")
        try:
            orch_leader.sqs_queue.send_message(queue_name=orch_leader.queue_name, message_dict=payload_inference)
        except Exception as e_inner:
            print(f"[TEST ERRORE CRITICO] Impossibile inviare il messaggio: {e_inner}")
            return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0}

        start_time = time.perf_counter()

        # Attendi la finestra di ripresa parziale (1..N-1 worker completati),
        # con fallback su "almeno 1 completato" se la finestra è troppo stretta.
        wait_timeout = ft_cfg.get("max_wait_for_inference_start_seconds", 60)
        expected_workers = getattr(orch_leader, "num_workers", None) or orch_leader._get_active_worker_count()
        poll_interval = ft_cfg.get("inference_poll_interval_seconds", 0.5)

        waited = _wait_for_inference_in_progress_aws(
            orch_leader.state_manager, job_id, timeout=wait_timeout, interval=poll_interval,
            min_workers=1, expected_workers=expected_workers, require_partial=True
        )
        if waited >= 0:
            print(f"[TEST KILLER] Finestra di ripresa parziale rilevata dopo {waited:.1f}s "
                  f"(>=1 worker completato su DynamoDB, non ancora tutti). Simulo il crash.")
        else:
            waited = _wait_for_inference_in_progress_aws(
                orch_leader.state_manager, job_id, timeout=wait_timeout, interval=poll_interval,
                min_workers=1, require_partial=False
            )
            if waited >= 0:
                print(f"[TEST KILLER] [WARN] Finestra parziale non colta: il crash cade su "
                      f"'inferenza avviata' dopo {waited:.1f}s. Failover comunque testato, ma la "
                      f"ripresa PARZIALE potrebbe non essere esercitata.")
            else:
                kill_delay = ft_cfg.get("docker_kill_delay_seconds", 2)
                print(f"[TEST KILLER] [WARN] Nessun segnale di inferenza entro {wait_timeout}s: "
                      f"ripiego su un ritardo fisso di {kill_delay}s. Verificare il risultato.")
                time.sleep(kill_delay)

        # Identifica e ferma il leader reale (rilettura al momento del kill,
        # può differire da un'eventuale lettura precedente).
        killed_leader_name = _get_current_leader_name_aws(orch_leader.state_manager)
        if not killed_leader_name:
            print(f"[TEST ERRORE] Nessun leader trovato sul lock '{_LOCK_KEY}': impossibile simulare il crash.")
            return {"status": "FAILED", "duration_seconds": round(time.perf_counter() - start_time, 2),
                    "error": "Nessun leader eletto su OrchestratorLocks."}

        ip_match = re.search(r"ip-(\d+-\d+-\d+-\d+)\.ec2\.internal", killed_leader_name)
        stopped = False
        if ip_match:
            # SENZA prefisso 'ip-': vedi nota in _resolve_ecs_task_arn_by_ip
            # sul formato atteso per il confronto con l'IP letto da ECS.
            ip_dashed = ip_match.group(1)
            try:
                ecs = boto3.client("ecs", region_name=region)
                target_arn = _resolve_ecs_task_arn_by_ip(ecs, cluster, service_name, ip_dashed)
                if target_arn:
                    print(f"\n[TEST TRIGGER] !!! SIMULAZIONE CRASH IMPREVISTO: fermo il task ECS del leader "
                          f"'{killed_leader_name}' ({target_arn.split('/')[-1]}) !!!")
                    ecs.stop_task(cluster=cluster, task=target_arn,
                                  reason="[TEST] Simulazione crash orchestratore leader durante inferenza (failover scenario)")
                    stopped = True
                else:
                    print(f"[TEST ERRORE] Nessun task ECS di '{service_name}' corrisponde all'IP del leader "
                          f"'{killed_leader_name}': impossibile fermarlo.")
            except Exception as e:
                print(f"[TEST ERRORE] Impossibile fermare il task ECS del leader: {e}")
        else:
            print(f"[TEST ERRORE] Formato nome leader inatteso ('{killed_leader_name}'): impossibile estrarne l'IP.")

        if not stopped:
            return {"status": "FAILED", "duration_seconds": round(time.perf_counter() - start_time, 2),
                    "error": "Impossibile identificare/fermare il task ECS del leader.",
                    "original_leader": killed_leader_name}

        # Monitoraggio fino al completamento — nessuna scorciatoia su lock/lease.
        failover_detection_margin = ft_cfg.get("failover_detection_margin_seconds", 90)
        max_timeout = ft_cfg.get("max_monitor_timeout_seconds", failover_detection_margin + 360)
        job_completed = False
        completed_by = None
        elapsed = 0
        check_interval = 3

        print(f"[TEST] Monitoraggio dello stato del Job (timeout: {max_timeout}s)...")
        while elapsed < max_timeout:
            time.sleep(check_interval)
            elapsed += check_interval
            try:
                state = orch_leader.state_manager.obtain_request(job_id)
                if state:
                    item_data = state.get("Item", state)
                    status = item_data.get("status")
                    current_owner = item_data.get("last_orchestrator") or item_data.get("orchestrator_id")
                    print(f"[TEST MONITOR] Stato: {status} | Gestore Attuale: {current_owner}")
                    if status == "COMPLETED":
                        job_completed = True
                        completed_by = current_owner
                        break
            except Exception as e:
                print(f"[TEST MONITOR WARNING] Errore lettura stato: {e}")

        duration = time.perf_counter() - start_time

        completed_by_killed_leader = bool(
            completed_by and killed_leader_name and completed_by == killed_leader_name
        )
        if completed_by_killed_leader:
            print(f"[TEST WARN] Il job risulta completato dallo stesso leader fermato ({completed_by}): "
                  f"il crash è probabilmente caduto a inferenza già conclusa. Failover NON esercitato.")

        if job_completed and not completed_by_killed_leader:
            test_status = "SUCCESS"
            print(f"\n[TEST PASSED] Failover reale completato! Job chiuso dallo standby subentrato ({completed_by}).")
        else:
            test_status = "FAILED"
            print(f"\n[TEST FAILED] Failover reale non verificato "
                  f"({'completato dal leader fermato' if completed_by_killed_leader else 'job non arrivato a COMPLETED entro il timeout'}).")

        return {
            "scenario_description": "Test di Failover REALE dell'Orchestratore durante l'inferenza sui 2 task ECS "
                                     "di orchestrator-service (leader fermato con ecs:StopTask, nessuna scorciatoia "
                                     "sul lock: recovery affidato al TTL naturale).",
            "status": test_status,
            "original_leader": killed_leader_name,
            "recovered_by": completed_by if (job_completed and not completed_by_killed_leader) else "None",
            "duration_seconds": round(duration, 2)
        }