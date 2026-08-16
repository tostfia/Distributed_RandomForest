import subprocess
import time
import threading
import os
import json
from src.testing.scenarios.base import BaseTestScenario
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator
import docker


def _wait_for_leadership(orch, timeout=15, interval=0.5) -> bool:
    """
    Attende (con timeout) che 'orch' risulti effettivamente leader, prima di
    procedere: senza questo controllo il test può inviare il job e simulare
    il crash mentre 'orch' non ha ancora davvero acquisito la leadership (o
    non l'acquisisce affatto), rendendo il "kill del leader" un'operazione
    su un processo che di fatto non stava facendo nulla.
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


class OrchestratorFailoverScenario(BaseTestScenario):
    """
    Copre lo Scenario: Failover dell'Orchestratore.
    Usa l'orchestratore nativo del TestEngine come Leader (Master-1) e 
    istanzia un secondo orchestratore di Standby (Master-2) per il subentro.
    """
 
    
    def run(self) -> dict:
     
        ft_cfg = self.config.get("fault_tolerance", {})
        
        
        
        # Identifichiamo il Leader di sistema 
        orch_leader = self.orchestrator
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
            kill_after_seconds = ft_cfg.get("kill_orchestrator_after_seconds", 270)
            print(f"[TEST KILLER] Lascio lavorare il leader per {kill_after_seconds} secondi prima del crash...")
            time.sleep(kill_after_seconds)
            
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
        if docker_env != "true":
            killer_thread = threading.Thread(target=kill_system_leader_target, name="KillerThread")
            killer_thread.daemon = True
            killer_thread.start()
        else: 
        
            client = docker.from_env()
            
            containers = client.containers.list(filters={
                "label": "com.docker.compose.service=orchestrator"
            })
            found = False
            for c in containers:

                if "orchestrator-1" in c.name:
                    print(f"[TEST TRIGGER] Disattivo la restart policy di {c.name} per evitare che risorga da solo...")
                    
                    c.update(restart_policy={"Name": "no"})
                    print(f"[TEST TRIGGER] Kill fisico del container: {c.name}")
                    c.kill()
                    found = True
                    break
            if not found:
                print("[TEST TRIGGER ERROR] Orchestratore-1 non trovato tra i container attivi.")
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

        kill_after_seconds = ft_cfg.get("kill_orchestrator_after_seconds", 120)
        failover_detection_margin = ft_cfg.get("failover_detection_margin_seconds", 90)
        max_timeout = ft_cfg.get("max_monitor_timeout_seconds", kill_after_seconds + failover_detection_margin + 60)
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