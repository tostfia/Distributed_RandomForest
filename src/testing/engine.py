import json
import os
import subprocess
import time

from src.testing.scenarios.fault import FaultToleranceScenario
from src.testing.scenarios.network import NetworkSimulationScenario
from src.testing.scenarios.performance import PerformanceAndMetricsScenario
from src.testing.scenarios.scalability import ScalabilityScenario



class TestEngine:
    """Engine principale che orchestra l'esecuzione di tutte le suite di test."""
    def __init__(self, config_path: str, orchestrator):
        self.config_path = config_path
        self.orchestrator = orchestrator
        self.config = self._load_config()
        self.global_reports = {}
        self.worker_processes = []

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[ERRORE ENGINE] File {self.config_path} non trovato. Uso config di fallback.")
            return {"selected_task": "classifier", "dataset_path": "synthetic/synthetic_dataset.csv"}
        
    def _start_local_workers(self):
        print("\n[ENGINE SYSTEM] Avvio automatico dei Worker in background...")
        num_workers = int(os.environ.get("NUM_WORKERS", 2))
        port_base = 18861
        for i in range(1, num_workers + 1):
            worker_name = f"Worker-Locale-{i:02d}"
            port = port_base + i -1
            print(f"[ENGINE SYSTEM] Avvio {worker_name} sulla porta {port}...")
            #si blocca su macchine senza gui gnome-terminal
            cmd = ["gnome-terminal", "--", "bash", "-c", f"python -m src.worker.main {worker_name} {port}; exec bash"]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.worker_processes.append(p)
        print(f"[ENGINE SYSTEM] Avvio dei {num_workers} Worker completato. Attendere 5 secondi per l'inizializzazione...")
        time.sleep(5)  # Attendere che i worker siano pronti

    def _cleanup_workers(self):
        print("\n[ENGINE SYSTEM] Arresto dei Worker in background...")
        for p in self.worker_processes:
            try: 
                p.terminate()
                p.wait(timeout=5)
            except Exception as e:
                p.kill()
        print("[ENGINE SYSTEM] Tutti i Worker sono stati arrestati.")


    def run_scenarios(self):
        print("\n==================================================")
        print("       AVVIO AUTOMATICO ENGINE DI TEST DI SISTEMA  ")
        print("==================================================")
        self._start_local_workers()
        try:
            print("Seleziona lo scenario da eseguire:")
            print("1. Performance e Metriche")
            print("2. Scalabilità")
            print("3. Simulazione di Rete")
            print("4. Tolleranza ai Guasti")
            valid_options = ["1", "2", "3", "4", "all"]
            while True:
                user_choice = input("Scelta (1-4, o 'all' per eseguire tutti): ").strip().lower()
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
            if config_mode != "all":
                self._print_final_summary()
        finally:
            self._cleanup_workers()
            
    
    def _run_all_scenarios(self):
        print("\n--- Esecuzione di tutti gli scenari di test ---")

        # Scenario 1 & 4
        perf_scenario = PerformanceAndMetricsScenario(self.config, self.orchestrator)
        self.global_reports["performance_and_metrics"] = perf_scenario.run()
        
        # Scenario 2
        scal_scenario = ScalabilityScenario(self.config, self.orchestrator)
        self.global_reports["scalability"] = scal_scenario.run()
        
        # Scenario 3
        net_scenario = NetworkSimulationScenario(self.config, self.orchestrator)
        self.global_reports["network_simulation"] = net_scenario.run()
        
        # Scenario 5
        fault_scenario = FaultToleranceScenario(self.config, self.orchestrator)
        self.global_reports["fault_tolerance"] = fault_scenario.run()
        
        self._print_final_summary()

    def _print_final_summary(self):
        print("\n==================================================")
        print("          SUMMARY REPORT FINALE DEI TEST          ")
        print("==================================================")
        print(json.dumps(self.global_reports, indent=2))
        print("==================================================")

        output_dir = "./.local_storage"
        output_path = os.path.join(output_dir, "test_report.json")
        
        try:
            
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.global_reports, f, indent=2)
                
            print(f"[ENGINE SYSTEM] Report delle metriche salvato in: '{output_path}'")
        except IOError as e:
            print(f"[ENGINE ERRORE] Impossibile salvare il report su disco: {e}")