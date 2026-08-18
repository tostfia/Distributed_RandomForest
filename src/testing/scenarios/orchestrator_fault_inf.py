import subprocess
import time
import threading
import os
import re
import json
from src.testing.scenarios.base import BaseTestScenario
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator
import docker


def _wait_for_leadership(orch, timeout=15, interval=0.5) -> bool:
    """
    Attende (con timeout) che 'orch' risulti effettivamente leader, prima di
    procedere: senza questo controllo il test può inviare il job di inferenza
    e simulare il crash mentre 'orch' non ha ancora davvero acquisito la
    leadership (o non l'acquisisce affatto), rendendo il "kill del leader"
    un'operazione su un processo che di fatto non stava facendo nulla.
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

def _wait_for_inference_in_progress(job_id, timeout=60, interval=0.2) -> float:
    """
    Attende (con timeout) che l'inferenza distribuita sia DAVVERO in corso,
    rilevando la comparsa del checkpoint di inferenza su disco con almeno un
    chunk completato.

    Perché serve: a differenza del training (che scrive alberi_addestrati>0
    nello StateManager, un segnale condiviso e leggibile), l'inferenza aggiorna
    lo stato del job solo a "COMPLETED", a fine lavoro. Un ritardo fisso prima
    del kill è quindi una scommessa: troppo presto → nessun chunk salvato, lo
    standby rifà tutto da zero (non si testa la ripresa da checkpoint); troppo
    tardi → il leader ha già finito e marcato COMPLETED, e il failover non
    viene testato affatto (falso positivo).

    L'orchestratore, però, dopo OGNI chunk di inferenza completato da un worker,
    salva la lista cumulativa dei chunk in
    './.local_storage/inference_chunks_{job_id}.pkl' (vedi
    _get_inference_checkpoint_path / _execute_inference_step in centralized.py).
    Quel file — visibile al test via bind mount — è il segnale reale che
    l'inferenza è iniziata ma non è detto sia finita: appena esiste ed è
    leggibile (>=1 chunk), siamo nella finestra giusta per simulare il crash.

    Ritorna i secondi attesi, o -1.0 se scade il timeout senza vedere il
    checkpoint (l'inferenza potrebbe essere già finita, o non essere partita).
    Applicabile solo in ambiente 'local' (checkpoint su filesystem locale).
    """
    checkpoint_path = f"./.local_storage/inference_chunks_{job_id}.pkl"
    waited = 0.0
    while waited < timeout:
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
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


class InferenceOrchestratorFaultScenario(BaseTestScenario):
    """
    Copre lo Scenario: Failover dell'Orchestratore durante la fase di inferenza.
    Usa l'orchestratore nativo del TestEngine come Leader (Master-1) e 
    istanzia un secondo orchestratore di Standby (Master-2) per il subentro.
    """

    def run(self) -> dict:

        ft_cfg = self.config.get("inference_orchestrator_failover", {})
        task_type = self.config.get("selected_task", "classifier")
        if task_type == "classifier":
            target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
        else:
            target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)

        orch_leader = self.orchestrator
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
            # Attendiamo un segnale REALE che l'inferenza è in corso — la
            # comparsa del checkpoint di inferenza su disco con >=1 chunk — invece
            # di un ritardo fisso "a scommessa". Così il crash cade sempre nella
            # finestra utile: dopo che almeno un chunk è stato prodotto (c'è lavoro
            # reale da recuperare) ma prima che il job sia COMPLETED (il failover
            # viene davvero esercitato). Vedi _wait_for_inference_in_progress.
            wait_timeout = ft_cfg.get("max_wait_for_inference_start_seconds", 60)
            waited = _wait_for_inference_in_progress(job_id, timeout=wait_timeout)
            if waited < 0:
                # Fallback: nessun checkpoint comparso entro il timeout. Può
                # succedere se l'inferenza è così rapida da completarsi prima di
                # essere osservata, o se non è partita. Ripieghiamo sul vecchio
                # ritardo fisso e segnaliamo che il test potrebbe non aver colto
                # un failover genuino.
                kill_delay = ft_cfg.get("docker_kill_delay_seconds", 1)
                print(f"[TEST KILLER] [WARN] Nessun checkpoint di inferenza rilevato entro {wait_timeout}s: "
                      f"ripiego su un ritardo fisso di {kill_delay}s. Il crash potrebbe cadere fuori dalla "
                      f"finestra utile (inferenza non ancora avviata o già conclusa): verificare il risultato.")
                time.sleep(kill_delay)
            else:
                print(f"[TEST KILLER] Inferenza in corso rilevata dopo {waited:.1f}s "
                      f"(checkpoint con lavoro reale presente). Simulo il crash immediatamente.")

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
        # NIENTE fallback glob "file più recente in ./saved_models/": in una
        # sessione di test con più job può facilmente esistere un modello
        # PIÙ RECENTE ma di un job completamente estraneo, che farebbe
        # dichiarare SUCCESS un failover che in realtà non è mai avvenuto.
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