import json
import os
import subprocess
import time

from src.master.orchestrator.centralized import CentralizedOrchestrator
from src.master.orchestrator.federated import FederatedOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.testing.scenarios.fault import FaultToleranceScenario
from src.testing.scenarios.network import NetworkSimulationScenario
from src.testing.scenarios.performance import PerformanceAndMetricsScenario
from src.testing.scenarios.scalability import ScalabilityScenario
from src.testing.scenarios.orchestrator_fault import OrchestratorFailoverScenario


CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_config.json")

class TestEngine:
    """Engine principale che orchestra l'esecuzione di tutte le suite di test."""
    def __init__(self, mode: str , env: str ):
        self.config_path = CONFIG_FILE_PATH
        self.mode = mode
        self.env = env
        self.config = self._load_config()
        self.global_reports = {}
        self.orchestrator = None
        self.worker_processes = []

        self._initialize_infrastructure()
        
    def _load_config(self) -> dict:

        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[ERRORE ENGINE] File {self.config_path} non trovato. Uso config di fallback.")
            return {"selected_task": "classifier", "dataset_path": "synthetic/synthetic_dataset.csv"}

    def _initialize_infrastructure(self):
     
        self.orchestrator = CentralizedOrchestrator(orchestrator_name= "orchestrator-centralizzato-testing") if self.mode == "centralized" else FederatedOrchestrator(orchestrator_name= "orchestrator-federato-testing")   
        self._start_local_workers()

    def _start_local_workers(self):
        docker = os.environ.get("RUNNING_IN_DOCKER")
        if docker == "true":
            print("[ENGINE] Ambiente DOCKER rilevato: avviati già i worker come container...")
            return
        else:
            num_workers = int(os.environ.get("NUM_WORKERS", 2))
            port_base = 18861
            print("[ENGINE] Avvio dei worker locali...")
            for i in range(1, num_workers + 1):
                worker_name = f"Worker-Locale-{i:02d}"
                port = port_base + i-1
                print(f"[ENGINE] Avvio {worker_name} sulla porta {port}...")
                cmd = ["python", "-m", "src.worker.main", worker_name, str(port)]
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.worker_processes.append(p)
            time.sleep(1)  # Simula il tempo di avvio dei worker
            print("[ENGINE] Worker locali avviati.")    

    def _cleanup_workers(self):
        print("[ENGINE] Pulizia dei worker locali...")
        for p in self.worker_processes:
            p.terminate()
            p.wait()
        print("[ENGINE] Worker locali terminati.")
    

    def run_scenarios(self):
        print("\n==================================================")
        print("       AVVIO AUTOMATICO ENGINE DI TEST DI SISTEMA  ")
        print("==================================================")
     
        try:
            print("Seleziona lo scenario da eseguire:")
            print("1. Performance e Metriche")
            print("2. Scalabilità")
            print("3. Simulazione di Rete")
            print("4. Tolleranza ai Guasti")
            print("5. Failover dell'Orchestratore")
            valid_options = ["1", "2", "3", "4","5", "all"]
            while True:
                user_choice = input("Scelta (1-5, o 'all' per eseguire tutti): ").strip().lower()
                if user_choice in valid_options:
                    config_mode = user_choice
                    break
                print("[ERRORE] Opzione non valida. Riprova.")
            if config_mode == "all":
                self._run_all_scenarios()
            elif config_mode == "1":
                perf_scenario = PerformanceAndMetricsScenario(self.config, self.orchestrator)
                self.global_reports["performance_and_metrics"] = perf_scenario.run()
            elif config_mode == "2":
                scal_scenario = ScalabilityScenario(self.config, self.orchestrator)
                self.global_reports["scalability"] = scal_scenario.run()
            elif config_mode == "3":
                net_scenario = NetworkSimulationScenario(self.config, self.orchestrator)
                self.global_reports["network_simulation"] = net_scenario.run()
            elif config_mode == "4":
                fault_scenario = FaultToleranceScenario(self.config, self.orchestrator)
                self.global_reports["fault_tolerance"] = fault_scenario.run()
            elif config_mode == "5":
                orchestrator_fault_scenario = OrchestratorFailoverScenario(self.config, self.orchestrator)
                self.global_reports["orchestrator_failover"] = orchestrator_fault_scenario.run()
            if config_mode != "all":
                self._print_final_summary()
        finally:
            docker = os.environ.get("RUNNING_IN_DOCKER")
            if docker != "true":
                print("[ENGINE] Pulizia dei worker locali...")
                self._cleanup_workers()
            
            
    
    def _run_all_scenarios(self):
        print("\n--- Esecuzione di tutti gli scenari di test ---")

        # Scenario 1 
        perf_scenario = PerformanceAndMetricsScenario(self.config, self.orchestrator)
        self.global_reports["performance_and_metrics"] = perf_scenario.run()
        
        # Scenario 2
        scal_scenario = ScalabilityScenario(self.config, self.orchestrator)
        self.global_reports["scalability"] = scal_scenario.run()
        
        # Scenario 3
        net_scenario = NetworkSimulationScenario(self.config, self.orchestrator)
        self.global_reports["network_simulation"] = net_scenario.run()
        
        # Scenario 4
        fault_scenario = FaultToleranceScenario(self.config, self.orchestrator)
        self.global_reports["fault_tolerance"] = fault_scenario.run()

        #Scenario 5
        orchestrator_fault_scenario = OrchestratorFailoverScenario(self.config, self.orchestrator)
        self.global_reports["orchestrator_failover"] = orchestrator_fault_scenario.run()

        
        self._print_final_summary()

    def _print_final_summary(self):
        print("\n==================================================")
        print("          SUMMARY REPORT FINALE DEI TEST          ")
        print("==================================================")
        print(json.dumps(self.global_reports, indent=2))
        print("==================================================")

        output_dir = "./test_reports"
        test_name = "all_tests" if len(self.global_reports) != 1 else next(iter(self.global_reports.keys()))
        output_path = os.path.join(output_dir, f"test_report_{test_name}.json")
        
        try:
            
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.global_reports, f, indent=2)
                
            print(f"[ENGINE SYSTEM] Report delle metriche salvato in: '{output_path}'")
        except IOError as e:
            print(f"[ENGINE ERRORE] Impossibile salvare il report su disco: {e}")


if __name__ == "__main__":
    # Inizializza il motore (ad esempio con valori di default o letti da env)
    engine = TestEngine(mode="federated", env="local")
    # Avvia l'interazione
    engine.run_scenarios()
