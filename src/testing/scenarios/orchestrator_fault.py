import subprocess
import time
import threading
import os
from src.testing.scenarios.base import BaseTestScenario
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator
import docker


class OrchestratorFailoverScenario(BaseTestScenario):
    """
    Copre lo Scenario: Failover dell'Orchestratore.
    Usa l'orchestratore nativo del TestEngine come Leader (Master-1) e 
    istanzia un secondo orchestratore di Standby (Master-2) per il subentro.
    """
 

    def run(self) -> dict:
     
        ft_cfg = self.config.get("fault_tolerance", {})
        target_alberi = 60
        expected_min = ft_cfg.get("expected_min_trees", 60)
        
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
                orch_standby = FederatedOrchestrator(orchestrator_name="Master-2-Standby")
                orch_standby.queue_name = target_queue
            else:
                orch_standby = CentralizedOrchestrator(orchestrator_name="Master-2-Standby")
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
            standby_thread = threading.Thread(target=orch_standby.start, name="StandbyThread")
            standby_thread.daemon = True
            standby_thread.start()
            
            time.sleep(2)  # Diamo tempo allo standby di assestarsi
           

        

        # 3. Generazione del Payload del Job
        job_id = f"test_orch_failover_{int(time.time())}"
        payload = {
            "job_id": job_id,
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {"n_estimators": target_alberi, "max_depth": 5, "tree_type": self.config["selected_task"]}
        }

        # 4. Invio del Job sulla coda standard gestita da Sofia
        print(f"[TEST] Invio del Job {job_id[:8]} alla coda '{orch_leader.queue_name}'...")
        try:
            orch_leader.sqs_queue.send_message(orch_leader.queue_name, payload)
        except TypeError:
            try:
                orch_leader.sqs_queue.send_message(queue_name=orch_leader.queue_name, message=payload)
            except Exception as e_inner:
                print(f"[TEST ERRORE CRITICO] Impossibile inviare il messaggio: {e_inner}")
                return {"status": "FAILED", "trees_built": 0, "duration_seconds": 0}

        start_time = time.perf_counter()

        # 5. Thread Killer: Abbate l'orchestratore di sistema Sofia a metà addestramento
        def kill_system_leader_target():
            kill_after_seconds = ft_cfg.get("kill_orchestrator_after_seconds")
            print(f"[TEST KILLER] Lascio lavorare il leader per {kill_after_seconds} secondi prima del crash...")
            time.sleep(kill_after_seconds)
            
            print(f"\n[TEST TRIGGER] !!! SIMULAZIONE CRASH IMPREVISTO DI {orch_leader.orchestrator_name.upper()} !!!")
            
            # Blocchiamo l'heartbeat loop del leader di sistema
            orch_leader._stop_heartbeat.set()
            
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
            try:
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
                
                print(f"[TEST TRIGGER] Container Leader terminato da Docker API.")
            except Exception as e:
                print(f"[TEST TRIGGER ERROR] Impossibile terminare il container: {e}")

        # 6. Monitoraggio dello stato del Job gestito dal subentrante Master-2
        # Il timeout deve coprire: tempo di lavoro pre-crash + tempo di detection del
        # leader morto da parte dello standby + tempo di completamento dei round residui.
        # Aggiungiamo un margine di sicurezza esplicito invece di un numero "a caso".
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
        job_id = orch_leader.current_job_id
        path_modello_dinamico = f"./saved_models/fed_model_{job_id}.pkl"

        if not job_id:
            import glob
            list_of_files = glob.glob('./saved_models/fed_model_*.pkl')
            path_modello_dinamico = max(list_of_files, key=os.path.getctime) if list_of_files else None

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