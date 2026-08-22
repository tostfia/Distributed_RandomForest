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
    Unisce il blocco di config 'key' (es. 'orchestrator_failover') con
    l'eventuale override AWS-specifico in
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


def _wait_for_job_processing(state_manager, job_id, timeout=600, interval=0.3) -> float:
    """
    Attende (con timeout) che i worker abbiano davvero completato e
    checkpointato almeno un chunk di alberi (alberi_addestrati > 0).

    NOTA: status=="PROCESSING" da solo NON basta — viene scritto subito dopo
    che il job viene reclamato, PRIMA che l'ETL (che può durare 130-260s sul
    dataset reale) e il primo giro di RPC ai worker siano anche solo iniziati.
    Usare solo quel segnale farebbe scattare il kill quando zero alberi sono
    stati prodotti, lasciando lo standby senza alcun checkpoint reale da cui
    ripartire — esattamente lo scenario che NON vogliamo testare.

    Il timeout di default (400s) è dimensionato per coprire l'intero ETL più
    il tempo del primo round di training, non solo l'acquisizione del lock.

    A differenza di chunk_sent_event (in-process, inutilizzabile quando il
    training avviene in un container/task separato), questo segnale è letto
    dallo stato condiviso e funziona identicamente in locale, Docker e AWS.
    Ritorna i secondi di attesa effettivi (utile per log), o -1.0 se il
    timeout scade prima che risulti del progresso reale.
    """
    waited = 0.0
    while waited < timeout:
        try:
            state = state_manager.obtain_request(job_id)
            if state:
                item_data = state.get("Item", state)
                if item_data.get("status") == "PROCESSING" and item_data.get("alberi_addestrati", 0) > 0:
                    return waited
        except Exception:
            pass
        time.sleep(interval)
        waited += interval
    return -1.0


def _wait_for_leadership(orch, timeout=15, interval=0.5) -> bool:
    """
    Attende (con timeout) che 'orch' risulti effettivamente leader, prima di
    procedere: senza questo controllo il test può inviare il job e simulare
    il crash mentre 'orch' non ha ancora davvero acquisito la leadership (o
    non l'acquisisce affatto), rendendo il "kill del leader" un'operazione
    su un processo che di fatto non stava facendo nulla.

    Usata SOLO dai rami locale/Docker Compose, dove il test-engine istanzia
    lui stesso Leader+Standby in-process. Il ramo AWS (vedi
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
            # Nessun getter dedicato esposto dallo state_manager per leggere il
            # leader corrente: un tentativo di acquisizione è idempotente se
            # 'orch' è già lui stesso il leader, quindi è un modo sicuro per
            # verificarlo senza scavalcare un altro leader legittimo.
            try:
                if orch._try_acquire_leadership():
                    return True
            except Exception:
                pass
        time.sleep(interval)
        waited += interval
    return False


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
    LEADER, vanificando il test: il job continuerebbe indisturbato sul
    leader originale e il test riporterebbe SUCCESS senza aver testato
    alcun failover reale.

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
    worker — vedi 'Worker-Fargate-...-ip-172-31-70-125.ec2.internal' nei log).
    Ritorna l'ARN del task, o None se non trovato.

    NOTA 'ip_dashed': deve essere SENZA il prefisso 'ip-' (es. '172-31-70-125'),
    perché va confrontato con 'ip.replace(".", "-")' qui sotto, che produce un
    IP puro senza prefisso (es. '172.31.70.125' -> '172-31-70-125'). Passare un
    valore con il prefisso 'ip-' (com'era in una versione precedente) fa
    fallire SEMPRE il confronto, anche quando il task cercato esiste davvero.
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


class OrchestratorFailoverScenario(BaseTestScenario):
    """
    Copre lo Scenario: Failover dell'Orchestratore.

    Su locale/Docker Compose: usa l'orchestratore nativo del TestEngine come
    Leader (Master-1) e istanzia un secondo orchestratore di Standby
    (Master-2) per il subentro — invariato rispetto alla versione originale.

    Su AWS: NON istanzia nulla in-process. Usa il vero orchestrator-service
    già dispiegato (2 task ECS, Leader+Standby sempre attivi) — vedi
    _run_aws_real_failover per il dettaglio.
    """


    def run(self) -> dict:

        # BUGFIX: leggeva erroneamente il blocco 'fault_tolerance' (parametri
        # dello scenario 4, Guasto Worker), che non contiene nessuna delle tre
        # chiavi usate da QUESTO scenario. Il risultato era che
        # max_wait_for_training_start_seconds/failover_detection_margin_seconds/
        # max_monitor_timeout_seconds ricadevano sempre sui default hardcoded
        # (400/90/790) sotto, IGNORANDO silenziosamente qualunque valore
        # impostato in test_config.json sotto 'orchestrator_failover'.
        ft_cfg = _merge_aws_overrides(self.config, "orchestrator_failover")

        orch_leader = self.orchestrator
        environment = getattr(orch_leader, "environment", "local")

        # AWS: percorso completamente separato, vedi _run_aws_real_failover.
        # Non condivide nulla con i rami Docker/locale sotto (che spawnano
        # orchestratori in-process, cosa che su AWS non ha senso: il vero
        # orchestrator-service è già lì, con 2 task reali).
        if environment == "aws":
            return self._run_aws_real_failover(orch_leader, ft_cfg)

        print(f"\n--- [TEST] Failover dell'Orchestratore con Leader di Sistema: '{orch_leader.orchestrator_name}' ---")
        orchestrator_type = os.environ.get("SYS_MODE", "centralized")
        target_queue = orch_leader.queue_name
        docker_env = os.environ.get("RUNNING_IN_DOCKER")

        # 1. Istanziamo solo l'orchestratore di Standby (Master-2), speculare al leader
        if docker_env == "true":
            print("[TEST] Ambiente DOCKER rilevato: avviato già Master-2 come container...")
            if orchestrator_type == "federated":
                orch_standby = FederatedOrchestrator(orchestrator_name="distributed_randomforest-orchestrator-2")
            else:
                orch_standby = CentralizedOrchestrator(orchestrator_name="distributed_randomforest-orchestrator-2")
            orch_standby.queue_name = target_queue
        else:
             # La coda di messaggi condivisa tra i due orchestratori
            if orchestrator_type == "federated":
                print("[TEST] Istanzio l'Orchestratore FEDERATO di Standby (Master-2)...")
                orch_standby = FederatedOrchestrator(orchestrator_name="Master-2-Standby")
                orch_standby.queue_name = target_queue
            else:
                print("[TEST] Istanzio l'Orchestratore CENTRALIZZATI di Standby (Master-2)...")
                orch_standby = CentralizedOrchestrator(orchestrator_name="Master-2-Standby")
                orch_standby.queue_name = target_queue

            # In locale (non-Docker) 'orch_leader' non è un processo esterno già
            # attivo: senza avviare qui il suo loop (.start()) non acquisirebbe
            # mai la leadership, e "uccidere il leader" più sotto agirebbe su un
            # orchestratore che di fatto non stava processando nulla. Lo avviamo
            # PRIMA dello standby e attendiamo che diventi davvero leader.
            leader_thread = threading.Thread(target=orch_leader.start, name="LeaderThread")
            leader_thread.daemon = True
            leader_thread.start()

            if not _wait_for_leadership(orch_leader, timeout=15):
                print(f"[TEST ERRORE] '{orch_leader.orchestrator_name}' non ha acquisito la leadership entro il timeout: "
                      f"il test non può simulare un failover credibile.")
                return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0,
                        "error": "Il leader designato non ha acquisito la leadership entro il timeout."}
            print(f"[TEST] '{orch_leader.orchestrator_name}' ha acquisito la leadership. Avvio dello Standby...")

            standby_thread = threading.Thread(target=orch_standby.start, name="StandbyThread")
            standby_thread.daemon = True
            standby_thread.start()

            time.sleep(2)  # Diamo tempo allo standby di assestarsi



        # 3. Generazione del Payload del Job
        job_id = f"test_orch_failover_{int(time.time())}"
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        payload = {
            "job_id": job_id,
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": hp
            }


        # 4. Invio del Job sulla coda standard gestita da Sofia
        print(f"[TEST] Invio del Job {job_id[:8]} alla coda '{orch_leader.queue_name}'...")
        try:
            orch_leader.sqs_queue.send_message(orch_leader.queue_name, payload)
        except TypeError:
            try:
                orch_leader.sqs_queue.send_message(queue_name=orch_leader.queue_name,message_dict=payload)
            except Exception as e_inner:
                print(f"[TEST ERRORE CRITICO] Impossibile inviare il messaggio: {e_inner}")
                return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0}

        start_time = time.perf_counter()

        # 5. Thread Killer
        def _simulate_backend_unreachable(orch):
            """
            Simula la perdita di accesso al coordination backend (DynamoDB/mock)
            da parte di 'orch', come avverrebbe in un vero crash o in una
            partizione di rete: la sua prossima chiamata a try_claim_job() —
            fatta periodicamente dal ciclo di _process_job già in corso, ad ogni
            round di alberi — fallirà, facendo scattare l'abort già previsto dal
            codice (MessageOwnershipLostError). Senza questo, il thread che sta
            elaborando il job continuerebbe a lavorare indisturbato in background
            nonostante il "crash" simulato, mantenendo la lease e impedendo allo
            standby di subentrare per davvero.
            """
            def _denied(*args, **kwargs):
                return False
            orch.state_manager.try_claim_job = _denied

        def kill_system_leader_target():
            timeout = ft_cfg.get("max_wait_for_training_start_seconds", 400)
            waited = _wait_for_job_processing(orch_leader.state_manager, job_id, timeout=timeout)
            if waited < 0:
                print(f"[TEST WARN] Timeout di {timeout}s raggiunto senza che risultasse alcun albero completato. "
                      f"Procedo comunque a simulare il guasto (nessun checkpoint reale da recuperare).")
            else:
                print(f"[TEST KILLER] Primo checkpoint reale rilevato dopo {waited:.1f}s: i worker hanno "
                      f"prodotto lavoro concreto. Simulo il crash immediatamente.")

            print(f"\n[TEST TRIGGER] !!! SIMULAZIONE CRASH IMPREVISTO DI {orch_leader.orchestrator_name.upper()} !!!")

            # Blocchiamo l'heartbeat loop PRIMA di toccare il lock: se lo
            # rimuovessimo con l'heartbeat ancora vivo, quest'ultimo lo
            # ricreerebbe da solo entro ~10s (_refresh_leadership_lock non
            # distingue "l'ho perso io" da "il file manca, lo riscrivo").
            orch_leader._stop_heartbeat.set()

            # Il lavoro già in corso (se presente) smette di essere "silenzioso":
            # al prossimo controllo di lease lo rileverà e abortirà da solo.
            _simulate_backend_unreachable(orch_leader)

            # Il patch sopra impedisce solo AL LEADER di riconquistare la lease,
            # ma il lock reale su JobLocks resta valido fino al suo TTL naturale
            # (300s, più lungo dell'intero timeout di monitoraggio del test): senza
            # rilasciarlo esplicitamente qui, lo standby otterrebbe sempre
            # CLAIM FAILED finché quel TTL non scade da solo.
            try:
                orch_leader.state_manager.release_job_lease(job_id, orch_leader.orchestrator_name)
                print(f"[TEST TRIGGER] Job lease di '{job_id[:8]}' rilasciata forzatamente.")
            except Exception:
                pass

            # Forziamo la rimozione del suo lock per svegliare immediatamente lo standby
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

        def kill_docker_leader_target():
            timeout = ft_cfg.get("max_wait_for_training_start_seconds", 400)
            waited = _wait_for_job_processing(orch_leader.state_manager, job_id, timeout=timeout)
            if waited < 0:
                print(f"[TEST WARN] Timeout di {timeout}s raggiunto senza che risultasse alcun albero completato "
                      f"(Docker). Procedo comunque a simulare il guasto (nessun checkpoint reale da recuperare).")
            else:
                print(f"[TEST KILLER] Primo checkpoint reale rilevato dopo {waited:.1f}s (Docker): i worker "
                      f"hanno prodotto lavoro concreto. Simulo il crash immediatamente.")

            print(f"\n[TEST TRIGGER] !!! SIMULAZIONE CRASH IMPREVISTO (DOCKER) !!!")

            client = docker.from_env()
            target_container = _resolve_leader_container(client)

            if target_container is None:
                # Fallback: non siamo riusciti a leggere il lock (assente o
                # corrotto). Come ultima risorsa proviamo comunque
                # 'orchestrator-1', ma segnaliamo chiaramente che potrebbe
                # essere il container sbagliato — meglio un test che grida
                # la propria incertezza che uno che dichiara SUCCESS avendo
                # ucciso lo standby per sbaglio.
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
                print(f"[TEST TRIGGER] Leader identificato dal lock: {target_container.name}. "
                      f"Disattivo la restart policy per evitare che risorga da solo...")
                target_container.update(restart_policy={"Name": "no"})
                print(f"[TEST TRIGGER] Kill fisico del container: {target_container.name}")
                target_container.kill()
            else:
                print("[TEST TRIGGER ERROR] Nessun container leader trovato tra i container attivi.")

            # Il container è morto per davvero (kill fisico): a differenza del
            # ramo non-Docker, qui non c'è un heartbeat in-process da fermare
            # prima — resta solo da ripulire il lock, che il container appena
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

            # NOTA: qui non possiamo forzare il rilascio della JOB LEASE (a
            # differenza del ramo non-Docker) perché il vero proprietario del
            # lock non è 'orch_leader' (l'handle di questo TestEngine) ma il
            # container 'orchestrator-1' con la sua identità interna — un
            # release_job_lease firmato da un orchestrator_name diverso dal
            # reale possessore fallisce silenziosamente. Bisogna quindi
            # dimensionare max_monitor_timeout_seconds includendo il naturale
            # TTL di 300s della lease (vedi try_claim_job in localstatemanager.py).
            print(f"[TEST TRIGGER] '{orch_leader.orchestrator_name}' [Docker] è stato neutralizzato.")

        if docker_env != "true":
            killer_thread = threading.Thread(target=kill_system_leader_target, name="KillerThread")
            killer_thread.daemon = True
            killer_thread.start()
        else:
            killer_thread = threading.Thread(target=kill_docker_leader_target, name="DockerKillerThread")
            killer_thread.daemon = True
            killer_thread.start()

        # Il kill ora scatta dopo il primo checkpoint reale (alberi_addestrati > 0),
        # non più al semplice status PROCESSING: il timeout di monitoraggio deve
        # coprire l'attesa per il primo checkpoint (che include l'intero ETL,
        # 130-260s sul dataset reale) + il recovery. Il recovery però è più
        # rapido di prima: l'ETL è già stato salvato su storage condiviso dal
        # leader originale, quindi lo standby lo salta ([SHORT-CIRCUIT ETL]) e
        # riparte solo dagli alberi mancanti.
        max_wait_for_training_start = ft_cfg.get("max_wait_for_training_start_seconds", 400)
        failover_detection_margin = ft_cfg.get("failover_detection_margin_seconds", 90)
        max_timeout = ft_cfg.get("max_monitor_timeout_seconds", max_wait_for_training_start + failover_detection_margin + 300)
        job_completed = False
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
                    current_owner = item_data.get("orchestrator_id")

                    print(f"[TEST MONITOR] Stato: {status} | Alberi a checkpoint: {trees_built} | Gestore Attuale: {current_owner}")

                    if status == "COMPLETED":
                        job_completed = True
                        break
            except Exception as e:
                print(f"[TEST MONITOR WARNING] Errore lettura stato: {e}")

        # Se il job è COMPLETED ma l'ultimo campo letto si è azzerato, riportiamo il
        # massimo visto durante il monitoraggio (che riflette il lavoro reale svolto).
        if job_completed and trees_built == 0:
            trees_built = max_trees_built_seen

        duration = time.perf_counter() - start_time
        # Il modello viene sempre salvato come "model_{job_id}.pkl" in entrambe le
        # modalità (vedi _resolve_model_path in centralized.py e federated.py).
        # NIENTE fallback glob "file più recente in ./saved_models/": in una
        # sessione di test con più job può facilmente esistere un modello
        # PIÙ RECENTE ma di un job completamente estraneo, che farebbe
        # dichiarare SUCCESS un failover che in realtà non è mai avvenuto.
        path_modello_dinamico = f"./saved_models/model_{job_id}.pkl"

        if job_completed or (trees_built == 0 and path_modello_dinamico and os.path.exists(path_modello_dinamico)):
            test_status = "SUCCESS"
            print(f"\n[TEST PASSED] Failover completato! Il modello {path_modello_dinamico} è stato generato.")
        else:
            test_status = "FAILED"
            print(f"\n[TEST FAILED] Failover fallito.")
        return {
            "scenario_description": "Test di Failover dell'Orchestratore con transizione della leadership dello stato su crash del Master.",
            "status": test_status,
            "original_leader": orch_leader.orchestrator_name,
            "recovered_by": "Master-2-Standby" if job_completed else "None",
            "duration_seconds": duration,
            "trees_built": trees_built
        }

    # ------------------------------------------------------------------ #
    # AWS: failover REALE sui 2 task ECS del orchestrator-service         #
    # ------------------------------------------------------------------ #

    def _run_aws_real_failover(self, orch_leader, ft_cfg) -> dict:
        """
        Failover REALE sui due task ECS del orchestrator-service già
        dispiegato (desired-count=2: un Leader + uno Standby, sempre attivi
        in produzione, indipendentemente da questo test). A differenza del
        ramo locale/Docker, qui non viene istanziato alcun orchestratore
        in-process: il test-engine si limita a
          1. inviare il job sulla coda SQS reale, che il leader reale (uno
             dei due task, già in ascolto) reclamerà;
          2. attendere il primo checkpoint reale (stesso segnale usato dagli
             altri rami: alberi_addestrati > 0 su DynamoDB);
          3. identificare QUALE dei due task è il leader leggendo il lock
             condiviso su DynamoDB (tabella OrchestratorLocks, campo
             'leader' — vedi dynamodb_aws.py/try_acquire_lock) e fermarlo
             fisicamente con ecs:StopTask;
          4. monitorare via DynamoDB fino al completamento, verificando che
             sia stato l'ALTRO task (lo standby sopravvissuto, o il suo
             rimpiazzo pianificato da ECS) a portare a termine il job.

        NESSUNA scorciatoia sul lock/lease: il recovery si affida agli stessi
        meccanismi già presenti in produzione. Il lock di leadership scade
        naturalmente dopo il suo TTL (180s, rinnovato ogni 10s dall'heartbeat
        — vedi BaseOrchestrator._heartbeat_loop/_refresh_leadership_lock)
        quando il leader smette di rinnovarlo; poi lo standby lo acquisisce e
        la job lease (TTL 300s) scade a sua volta, permettendo la ripresa via
        _perform_active_recovery. Il timeout di monitoraggio è dimensionato
        per coprire il caso peggiore di entrambe le attese in sequenza.
        """
        import boto3

        cluster, region, service_name = _resolve_aws_infra(self.config)

        print(f"\n--- [TEST] Failover REALE dell'Orchestratore su ECS "
              f"('{service_name}', cluster '{cluster}') ---")

        # 0. Verifica preliminare: deve già esserci un leader eletto tra i 2 task.
        leader_name = None
        for _ in range(15):
            leader_name = _get_current_leader_name_aws(orch_leader.state_manager)
            if leader_name:
                break
            time.sleep(1)
        if not leader_name:
            print(f"[TEST ERRORE] Nessun leader trovato sul lock '{_LOCK_KEY}' entro il timeout: "
                  f"verifica che '{service_name}' sia RUNNING con almeno 1 task attivo.")
            return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0,
                    "error": "Nessun leader eletto su OrchestratorLocks."}
        print(f"[TEST] Leader corrente identificato dal lock DynamoDB: '{leader_name}'.")

        # 1. Payload e invio del job sulla coda reale
        job_id = f"test_orch_failover_aws_{int(time.time())}"
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        payload = {
            "job_id": job_id,
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": hp,
        }
        print(f"[TEST] Invio del Job {job_id[:8]} alla coda '{orch_leader.queue_name}' "
              f"(lo reclamerà il leader reale su ECS)...")
        try:
            orch_leader.sqs_queue.send_message(queue_name=orch_leader.queue_name, message_dict=payload)
        except Exception as e_inner:
            print(f"[TEST ERRORE CRITICO] Impossibile inviare il messaggio: {e_inner}")
            return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0}

        start_time = time.perf_counter()

        # 2. Attendi il primo checkpoint reale, poi rileva e ferma il leader.
        wait_timeout = ft_cfg.get("max_wait_for_training_start_seconds", 600)
        waited = _wait_for_job_processing(orch_leader.state_manager, job_id, timeout=wait_timeout)
        if waited < 0:
            print(f"[TEST WARN] Timeout di {wait_timeout}s raggiunto senza che risultasse alcun albero completato. "
                  f"Procedo comunque a simulare il guasto (nessun checkpoint reale da recuperare).")
        else:
            print(f"[TEST KILLER] Primo checkpoint reale rilevato dopo {waited:.1f}s. Simulo il crash immediatamente.")

        # Rileggiamo il leader al momento del kill: potrebbe non essere più lo
        # stesso letto al punto 0 se nel frattempo c'è già stato un cambio.
        killed_leader_name = _get_current_leader_name_aws(orch_leader.state_manager) or leader_name
        ip_match = re.search(r"ip-(\d+-\d+-\d+-\d+)\.ec2\.internal", killed_leader_name or "")
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
                                  reason="[TEST] Simulazione crash orchestratore leader (failover scenario)")
                    stopped = True
                else:
                    print(f"[TEST ERRORE] Nessun task ECS di '{service_name}' corrisponde all'IP del leader "
                          f"'{killed_leader_name}': impossibile fermarlo.")
            except Exception as e:
                print(f"[TEST ERRORE] Impossibile fermare il task ECS del leader: {e}")
        else:
            print(f"[TEST ERRORE] Formato nome leader inatteso ('{killed_leader_name}'): impossibile estrarne l'IP.")

        if not stopped:
            return {"status": "FAILED", "trees_built": 0,
                    "duration_seconds": round(time.perf_counter() - start_time, 2),
                    "error": "Impossibile identificare/fermare il task ECS del leader.",
                    "original_leader": killed_leader_name}

        # 3. Monitoraggio fino al completamento — nessuna scorciatoia su
        # lock/lease, ci si affida al TTL naturale come in produzione.
        failover_detection_margin = ft_cfg.get("failover_detection_margin_seconds", 90)
        max_timeout = ft_cfg.get("max_monitor_timeout_seconds", wait_timeout + failover_detection_margin + 300)
        job_completed = False
        completed_by = None
        trees_built = 0
        max_trees_built_seen = 0
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
                    trees_built = item_data.get("alberi_addestrati", 0)
                    max_trees_built_seen = max(max_trees_built_seen, trees_built)
                    # AwsStateManager scrive 'last_orchestrator', non 'orchestrator_id'
                    # (vedi update_request_status/complete_request in awsstatemanager.py).
                    current_owner = item_data.get("last_orchestrator") or item_data.get("orchestrator_id")
                    print(f"[TEST MONITOR] Stato: {status} | Alberi a checkpoint: {trees_built} | Gestore Attuale: {current_owner}")
                    if status == "COMPLETED":
                        job_completed = True
                        completed_by = current_owner
                        break
            except Exception as e:
                print(f"[TEST MONITOR WARNING] Errore lettura stato: {e}")

        if job_completed and trees_built == 0:
            trees_built = max_trees_built_seen

        duration = time.perf_counter() - start_time

        # Il leader ucciso non torna mai in vita con la stessa identità: ECS
        # pianifica un task di RIMPIAZZO con un nuovo Task ID (quindi un nuovo
        # 'orchestrator_name', diverso da quello fermato) — un confronto per
        # uguaglianza esatta basta, a differenza del ramo Docker che deve
        # accontentarsi di un confronto per prefisso sull'hostname.
        completed_by_killed_leader = bool(
            completed_by and killed_leader_name and completed_by == killed_leader_name
        )
        if completed_by_killed_leader:
            print(f"[TEST WARN] Il job risulta completato dallo stesso leader fermato ({completed_by}): "
                  f"il crash è probabilmente caduto a job già concluso. Failover NON esercitato.")

        if job_completed and not completed_by_killed_leader:
            test_status = "SUCCESS"
            print(f"\n[TEST PASSED] Failover reale completato! Job chiuso dallo standby subentrato ({completed_by}).")
        else:
            test_status = "FAILED"
            print(f"\n[TEST FAILED] Failover reale non verificato "
                  f"({'completato dal leader fermato' if completed_by_killed_leader else 'job non arrivato a COMPLETED entro il timeout'}).")

        return {
            "scenario_description": "Test di Failover REALE dell'Orchestratore sui 2 task ECS di orchestrator-service "
                                     "(leader fermato con ecs:StopTask, nessuna scorciatoia sul lock: recovery affidato al TTL naturale).",
            "status": test_status,
            "original_leader": killed_leader_name,
            "recovered_by": completed_by if (job_completed and not completed_by_killed_leader) else "None",
            "duration_seconds": round(duration, 2),
            "trees_built": trees_built
        }