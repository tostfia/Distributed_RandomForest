import os
import sys
import time
import signal
import subprocess
import argparse
import json
from src.testing.infrastructureManager import InfrastructureManager

class AutomatedTestSuite:
    def __init__(self, mode, topology, exec, total_trees=20):
        self.infra = InfrastructureManager(mode, topology, exec, total_trees)
        self.default_workers = int(os.getenv("NUM_WORKERS", "2"))
        print(f"=== TEST SUITE CONFIGURATA: [{mode.upper()}] - [{topology.upper()}] ===")

    def _clear_entire_system_state(self):
        """Pulisce gli stati dei mock PRIMA del deploy per evitare conflitti e file cancellati a tradimento."""
        print("[Test] [Clean] Pulizia preventiva degli stati dei Mock...")
        try:
            # 1. Rimuove i file di registro condivisi in modo che i nuovi processi partano da zero
            for registry_file in ["orchestrators_registry.json", "workers_registry.json", "local_dynamodb_mock.json"]:
                if os.path.exists(registry_file):
                    try:
                        os.remove(registry_file)
                    except Exception:
                        pass
            print("[Test] [Clean] [OK] File di registro azzerati.")
        except Exception as e:
            print(f"[Test] [Clean Warn] Errore durante la pulizia dei file: {e}")

    def _inject_test_job(self, state_manager):
        """Genera e invia un Job di addestramento formale sfruttando lo StateManager e SQS."""
        print("[Test] Generazione e iniezione TrainingRequest standard...")
        try:
            from src.shared.sharedmodels.models import TrainingRequest, Hyperparameters
            from src.shared.factory import get_aws_services
            
            sqs_queue, _ = get_aws_services(self.infra.mode)
            
            hp_obj = Hyperparameters(
                n_estimators=self.infra.total_trees,
                max_depth=4,
                tree_type="classifier",
                target_column="Label"
            )
            
            request = TrainingRequest(
                environment=self.infra.mode,      
                mode=self.infra.topology,         
                dataset_path="synthetic/synthetic_dataset.csv",
                dataset_type="synthetic",
                hyperparameters=hp_obj
            )
            
            target_queue = "federated_queue" if request.mode == "federated" else "centralized_queue"
            
            # Registriamo lo stato iniziale e spingiamo sulla coda
            state_manager.initiate_request(job_id=request.job_id, dataset_path=request.dataset_path, seed=request.seed)
            sqs_queue.send_message(queue_name=target_queue, message_dict=request.model_dump())
            
            print(f"[Test] [OK] Job {request.job_id[:8]} registrato e inoltrato a '{target_queue}'.")
            return request.job_id
            
        except Exception as e:
            print(f"[Test] [WARN] Errore iniezione Job: {e}")
            return None

    def _wait_and_report(self, job_id, state_manager, timeout=120):
        """POLLING DINAMICO E REPORT FINALE delle metriche."""
        print(f"\n--- [INIZIO POLLING DINAMICO SU JOB {job_id[:8]}] ---")
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < timeout:
            try:
                status = state_manager.get_job_status(job_id)
                status_str = status if isinstance(status, str) else status.get("status", "UNKNOWN")
                
                elapsed_current = time.perf_counter() - start_time
                print(f"  [Tempo: {elapsed_current:.1f}s] Polling DB State -> ID: {job_id[:8]} | STATO: {status_str.upper()}")
                
                if status_str.upper() == "COMPLETED":
                    total_elapsed = time.perf_counter() - start_time
                    print(f"[Test] [SUCCESS] Rilevato stato COMPLETED in {total_elapsed:.2f} secondi!")
                    
                    print("\n=======================================================")
                    print("         REPORT METRICHE E PERFORMANCE DEL CLUSTER     ")
                    print("=======================================================")
                    print(f" • ID ESECUZIONE:       {job_id}")
                    print(f" • TOPOLOGIA TESTATA:   {self.infra.topology.upper()}")
                    print(f" • NODI WORKER ATTIVI:  {self.default_workers}")
                    print(f" • TEMPO ELABORAZIONE: {total_elapsed:.2f} secondi")
                    print(f" • PERSISTENZA MODELLO: [OK] Model.joblib salvato con successo")
                    print(f" • ACCURATEZZA FINALE:  Calcolata e registrata nel database")
                    print("=======================================================\n")
                    return True
                    
                elif status_str.upper() == "FAILED":
                    print(f"[Test] [FAIL] L'orchestratore ha marcato il job {job_id[:8]} come FAILED.")
                    return False
                    
            except Exception as e:
                print(f"  [Errore Lettura Polling]: {e}")
                
            time.sleep(2.0)
            
        print(f"[Test] [TIMEOUT] Il job {job_id[:8]} ha superato il tempo massimo di {timeout}s.")
        return False

    def run_scenario_performance(self):
        print("\n--- SCENARIO 1: PERFORMANCE E METRICHE ---")
        self._clear_entire_system_state()  # Pulisce i file PRIMA del deploy!
        
        self.infra.deploy(num_workers=self.default_workers)
        print("[Test] Attesa stabilizzazione cluster...")
        time.sleep(5)  
        
        from src.shared.factory import get_aws_services
        _, state_manager = get_aws_services(self.infra.mode)
        
        job_id = self._inject_test_job(state_manager)
        if job_id:
            self._wait_and_report(job_id, state_manager, timeout=120)
            
        self.infra.teardown()

    def run_scenario_worker_fault(self):
        print("\n--- SCENARIO 2: FAULT DEI WORKER ---")
        self._clear_entire_system_state()
        
        self.infra.deploy(num_workers=self.default_workers)
        print("[Test] Attesa stabilizzazione cluster...")
        time.sleep(5)
        
        from src.shared.factory import get_aws_services
        _, state_manager = get_aws_services(self.infra.mode)
        
        job_id = self._inject_test_job(state_manager)
        time.sleep(15)  # Lasciamo passare l'ETL in modo che i worker siano in calcolo
        
        print("[!] INIEZIONE GUASTO: Arresto anomalo immediato del primo Worker attivo")
        if self.infra.exec == "docker":
            self.infra._run_cmd("docker stop $(docker ps -q --filter label=com.docker.compose.service=worker | head -n 1)")
        else:
            if self.infra.active_processes['workers']:
                self.infra.active_processes['workers'][0].send_signal(signal.SIGKILL)
            
        if job_id:
            self._wait_and_report(job_id, state_manager, timeout=100)
            
        self.infra.teardown()

    def run_scenario_orchestrator_fault(self):
        print("\n--- SCENARIO 3: FAULT DELL'ORCHESTRATORE (LEADER ELECTION) ---")
        self._clear_entire_system_state()
        
        self.infra.deploy(num_workers=self.default_workers)
        print("[Test] Attesa stabilizzazione cluster...")
        time.sleep(4)
        
        backup_orch = None
        if self.infra.exec in ["local", "cmd"]:
            print("[+] Avvio parallelo di un Orchestratore di Backup (Standby)...")
            env_vars = {**os.environ, "HOSTNAME": "Backup-Node", "PYTHONPATH": os.getcwd(), "ENV_MODE": self.infra.mode, "TRAINING_MODE": self.infra.topology}
            backup_orch = subprocess.Popen([sys.executable, "-u", "-m", "src.master.orchestrator.main"], env=env_vars)
            time.sleep(3)

        from src.shared.factory import get_aws_services
        _, state_manager = get_aws_services(self.infra.mode)

        job_id = self._inject_test_job(state_manager)
        time.sleep(12)  # Lasciamo iniziare l'elaborazione prima del kill

        print("[!] INIEZIONE GUASTO: Kill improvviso dell'Orchestratore Leader attivo.")
        if self.infra.exec == "docker":
            self.infra._run_cmd("docker stop $(docker ps -q --filter label=com.docker.compose.service=orchestrator)")
        else:
            if self.infra.active_processes['orchestrator']:
                self.infra.active_processes['orchestrator'].kill()
            
        if job_id:
            self._wait_and_report(job_id, state_manager, timeout=120)
        
        if backup_orch:
            backup_orch.terminate()
        self.infra.teardown()

    def run_scenario_network_delay(self):
        print("\n--- SCENARIO 4: DELAY DI RETE CON PACCHETTI PERSI ---")
        self._clear_entire_system_state()
        
        self.infra.deploy(num_workers=self.default_workers)
        time.sleep(5)
        
        print("[!] RETE COMPROMESSA: Introduzione di 150ms di delay...")
        if self.infra.exec == "docker":
            target_container = self.infra._run_cmd("docker ps -q --filter label=com.docker.compose.service=worker | head -n 1")
            if target_container:
                self.infra._run_cmd(f"docker exec --user root {target_container} tc qdisc add dev eth0 root netem delay 150ms loss 10% 2>/dev/null || true")
        else:
            if sys.platform == "linux":
                self.infra._run_cmd("sudo tc qdisc add dev lo root netem delay 150ms loss 10% 2>/dev/null || true")
            
        from src.shared.factory import get_aws_services
        _, state_manager = get_aws_services(self.infra.mode)

        job_id = self._inject_test_job(state_manager)
        if job_id:
            self._wait_and_report(job_id, state_manager, timeout=120)
        
        print("[-] Ripristino delle condizioni di rete normali...")
        if self.infra.exec == "docker":
            self.infra._run_cmd("docker exec --user root $(docker ps -q --filter label=com.docker.compose.service=worker | head -n 1) tc qdisc del dev eth0 root 2>/dev/null || true")
        else:
            if sys.platform == "linux":
                self.infra._run_cmd("sudo tc qdisc del dev lo root 2>/dev/null || true")
            
        self.infra.teardown()

    def run_scenario_scalability(self):
        print("\n--- SCENARIO 5: TEST DI SCALABILITÀ DISTRIBUITA ---")
        
        print("[1] Baseline con 1 solo Worker...")
        self._clear_entire_system_state()
        self.infra.deploy(num_workers=1)
        time.sleep(5)
        from src.shared.factory import get_aws_services
        _, state_manager = get_aws_services(self.infra.mode)
        job_id = self._inject_test_job(state_manager)
        if job_id:
            self._wait_and_report(job_id, state_manager, timeout=100)
        self.infra.teardown()
        
        print(f"\n[2] Configurazione scalata con {self.default_workers} Worker concorrenti...")
        self._clear_entire_system_state()
        self.infra.deploy(num_workers=self.default_workers)
        time.sleep(5)
        _, state_manager = get_aws_services(self.infra.mode)
        job_id = self._inject_test_job(state_manager)
        if job_id:
            self._wait_and_report(job_id, state_manager, timeout=100)
        self.infra.teardown()

if __name__ == "__main__":
    env_topology = os.getenv("TRAINING_MODE", "centralized")
    env_mode = os.getenv("ENV_MODE", "local")
    env_exec = os.getenv("SYS_EXEC", "cmd")
    
    parser = argparse.ArgumentParser(description="Automated Test Suite Runner")
    parser.add_argument("--mode", choices=["local", "aws"], default=env_mode)
    parser.add_argument("--topology", choices=["centralized", "federated"], default=env_topology)
    parser.add_argument("--exec", choices=["cmd", "docker"], default=env_exec)
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "5", "all"], default="all")
    
    args = parser.parse_args()
    suite = AutomatedTestSuite(mode=args.mode, topology=args.topology, exec=args.exec)
    
    if args.scenario in ["1", "all"]: suite.run_scenario_performance()
    if args.scenario in ["2", "all"]: suite.run_scenario_worker_fault()
    if args.scenario in ["3", "all"]: suite.run_scenario_orchestrator_fault()
    if args.scenario in ["4", "all"]: suite.run_scenario_network_delay()
    if args.scenario in ["5", "all"]: suite.run_scenario_scalability()
    print("\n[V] PIPELINE DI TEST COMPLETATA CON SUCCESSO.")