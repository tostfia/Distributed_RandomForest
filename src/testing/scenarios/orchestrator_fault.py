import time
import threading
import os
from src.testing.scenarios.base import BaseTestScenario
from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator



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
        
        # 1. Istanziamo solo l'orchestratore di Standby (Master-2), speculare al leader
        orchestrator_type = self.config.get("orchestrator_type", "centralized").lower()
        if orchestrator_type == "federated":
            print("[TEST] Istanzio l'Orchestratore FEDERATO di Standby (Master-2)...")
            orch_standby = FederatedOrchestrator(orchestrator_name="Master-2-Standby")
        else:
            print("[TEST] Istanzio l'Orchestratore CENTRALIZZATI di Standby (Master-2)...")
            orch_standby = CentralizedOrchestrator(orchestrator_name="Master-2-Standby")

        # 2. Avviamo lo Standby (Master-2) in background. 
        # Troverà il lock occupato da Sofia ed entrerà in loop di attesa (Polling).
        def run_standby():
            print("[TEST] Avvio Master-2 (Standby) in ascolto del cluster...")
            orch_standby.start()

        standby_thread = threading.Thread(target=run_standby, name="StandbyThread")
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
            kill_after_seconds = ft_cfg.get("kill_orchestrator_after_seconds", 35)
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

        killer_thread = threading.Thread(target=kill_system_leader_target, name="KillerThread")
        killer_thread.daemon = True
        killer_thread.start()

        # 6. Monitoraggio dello stato del Job gestito dal subentrante Master-2
        max_timeout = 120  
        job_completed = False
        trees_built = 0
        elapsed = 0
        check_interval = 3
        
        print("[TEST] Monitoraggio dello stato del Job...")
        while elapsed < max_timeout:
            time.sleep(check_interval)
            elapsed += check_interval
            
            try:
                state = orch_standby.state_manager.obtain_request(job_id)
                if state:
                    item_data = state.get("Item", state)
                    status = item_data.get("status")
                    trees_built = item_data.get("alberi_addestrati", 0)
                    current_owner = item_data.get("orchestrator_id")
                    
                    print(f"[TEST MONITOR] Stato: {status} | Alberi a checkpoint: {trees_built} | Gestore Attuale: {current_owner}")
                    
                    if status == "COMPLETED":
                        job_completed = True
                        break
            except Exception as e:
                print(f"[TEST MONITOR WARNING] Errore lettura stato: {e}")

        duration = time.perf_counter() - start_time

        if job_completed and trees_built >= expected_min:
            test_status = "SUCCESS"
            print(f"\n[TEST PASSED] Failover completato! Master-2-Standby ha rilevato la morte di {orch_leader.orchestrator_name}, ha eseguito il ripristino e ha ultimato la foresta.")
        else:
            test_status = "FAILED"
            print(f"\n[TEST FAILED] Failover fallito o timeout raggiunto. Alberi completati: {trees_built}/{target_alberi}")

        return {
            "scenario_description": "Test di Failover dell'Orchestratore con transizione della leadership dello stato su crash del Master.",
            "status": test_status, 
            "original_leader": orch_leader.orchestrator_name, 
            "recovered_by": "Master-2-Standby" if job_completed else "None", 
            "duration_seconds": duration, 
            "trees_built": trees_built 
        }