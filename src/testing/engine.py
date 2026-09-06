import json
import os
import subprocess
import time

import boto3

from src.orchestrator.centralized import CentralizedOrchestrator
from src.orchestrator.federated import FederatedOrchestrator
from src.testing.scenarios.fault import FaultToleranceScenario
from src.testing.scenarios.network import NetworkSimulationScenario
from src.testing.scenarios.performance import PerformanceAndMetricsScenario
from src.testing.scenarios.scalability import ScalabilityScenario
from src.testing.scenarios.orchestrator_fault import OrchestratorFailoverScenario
from src.testing.scenarios.fault_inf import InferenceWorkerFaultScenario
from src.testing.scenarios.orchestrator_fault_inf import InferenceOrchestratorFaultScenario
from src.testing.scenarios.orchestrator_election_concurrency import OrchestratorElectionConcurrencyScenario


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
        self.n_samples_label = os.environ.get("SYNTHETIC_N_SAMPLES", "").strip()

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
        self._cleanup_stale_processing_jobs()
        self._start_local_workers()

    def _cleanup_stale_processing_jobs(self):
        """
        Gli scenario di test che chiamano _execute_training_step/_execute_inference_step
        DIRETTAMENTE (senza passare dalla coda SQS) non finalizzano mai il job a
        "COMPLETED" — quella transizione avviene solo dentro _process_job, che questi
        scenario bypassano di proposito per restare rapidi e isolati. Senza questa
        pulizia, un job di una sessione di test PRECEDENTE resta per sempre in
        "PROCESSING": al prossimo avvio, _perform_active_recovery() lo troverebbe e
        lo "riprenderebbe" da capo con iperparametri di default (nessun sidecar di
        metadati salvato, per lo stesso motivo), sprecando minuti di lavoro dei
        worker su un job estraneo prima ancora che il test attuale possa iniziare.
        """
        state_manager = getattr(self.orchestrator, "state_manager", None)
        if not state_manager or not hasattr(state_manager, "get_active_jobs"):
            return
        try:
            stale_job_ids = state_manager.get_active_jobs()
        except Exception as e:
            print(f"[ENGINE] [WARN] Impossibile leggere i job attivi per la pulizia iniziale: {e}")
            return

        cleaned = 0
        for job_id in stale_job_ids:
            try:
                existing_state = state_manager.obtain_request(job_id)
                if not existing_state:
                    continue
                item_data = existing_state.get("Item", existing_state)
                if item_data.get("status") != "PROCESSING":
                    continue
                state_manager.update_request_status(
                    job_id=job_id,
                    status="STALE_TEST_CLEANUP",
                    orchestrator_id="TEST-ENGINE-CLEANUP",
                    retries=0,
                    base_random_state=0,
                    alberi_addestrati=item_data.get("alberi_addestrati", 0),
                )
                cleaned += 1
            except Exception as e:
                print(f"[ENGINE] [WARN] Impossibile ripulire il job residuo {job_id[:8]}: {e}")

        if cleaned:
            print(f"[ENGINE] [CLEANUP] {cleaned} job residui da sessioni di test precedenti "
                  f"marcati come non-attivi (evitato un recovery indesiderato).")

    def _start_local_workers(self):
        docker = os.environ.get("RUNNING_IN_DOCKER")
        if docker == "true":
            # RUNNING_IN_DOCKER=true è corretto sia in Docker Compose locale sia
            # su AWS/ECS Fargate (i worker girano comunque dentro container), ma
            # va distinto nel messaggio per non confondere i due ambienti nei log
            # e nei report di test.
            if self.env == "aws":
                print("[ENGINE] Ambiente AWS/ECS rilevato: worker già avviati come task Fargate...")
            else:
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
                cmd = [
                    "python", "worker_supervisor.py", "--",
                    "python", "-m", "src.worker.main", worker_name, str(port), self.mode, self.env
                ]
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
            print("4. Guasto improvviso del Worker (addestramento)")
            print("5. Guasto improvviso del Worker (inferenza)")
            print("6. Failover dell'Orchestratore (addestramento)")
            print("7. Failover dell'Orchestratore (inferenza)")
            print("8. Elezione del Leader sotto Concorrenza (Safety)")
            print("9. Genera Grafici")
            valid_options = ["1", "2", "3", "4","5", "6", "7", "8", "9", "all"]
            # Bypass non-interattivo: se la variabile d'ambiente SCENARIO è
            # impostata (usato da run_test_engine_ecs.sh / task ECS one-off
            # senza terminale collegato all'avvio), la usiamo al posto del
            # prompt. Se non è impostata il comportamento resta identico a
            # oggi (chiede a terminale) — utile in locale/Docker/ECS Exec.
            env_choice = os.environ.get("SCENARIO", "").strip().lower()
            if env_choice in valid_options:
                config_mode = env_choice
                print(f"Scelta (da variabile d'ambiente SCENARIO): {config_mode}")
            else:
                while True:
                    user_choice = input("Scelta (1-9, o 'all' per eseguire tutti): ").strip().lower()
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
                fault_inf = InferenceWorkerFaultScenario(self.config, self.orchestrator)
                self.global_reports["inference_worker_fault"] = fault_inf.run()
            elif config_mode == "6":
                orchestrator_fault_scenario = OrchestratorFailoverScenario(self.config, self.orchestrator)
                self.global_reports["orchestrator_failover"] = orchestrator_fault_scenario.run()
            elif config_mode == "7":
                orchestrator_fault_inf = InferenceOrchestratorFaultScenario(self.config, self.orchestrator)
                self.global_reports["inference_orchestrator_failover"] = orchestrator_fault_inf.run()
            elif config_mode == "8":
                election_scenario = OrchestratorElectionConcurrencyScenario(self.config, self.orchestrator)
                self.global_reports["orchestrator_election_concurrency"] = election_scenario.run()
            elif config_mode == "9":
                from src.testing.plot_generator import PlotGenerator
                plotter = PlotGenerator()
                plotter.generate_all_plots()
            if config_mode not in ("all", "9"):
                self._print_final_summary()
        finally:
            docker = os.environ.get("RUNNING_IN_DOCKER")
            if docker != "true":
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
        fault_inf = InferenceWorkerFaultScenario(self.config, self.orchestrator)
        self.global_reports["inference_worker_fault"] = fault_inf.run()

        #Scenario 6
        orchestrator_fault_scenario = OrchestratorFailoverScenario(self.config, self.orchestrator)
        self.global_reports["orchestrator_failover"] = orchestrator_fault_scenario.run()

        #Scenario 7
        orchestrator_fault_inf = InferenceOrchestratorFaultScenario(self.config, self.orchestrator)
        self.global_reports["inference_orchestrator_failover"] = orchestrator_fault_inf.run()

        #Scenario 9 (Elezione del Leader sotto Concorrenza - Safety)
        election_scenario = OrchestratorElectionConcurrencyScenario(self.config, self.orchestrator)
        self.global_reports["orchestrator_election_concurrency"] = election_scenario.run()

        self._print_final_summary()

    def _print_final_summary(self):
        print("\n==================================================")
        print("          SUMMARY REPORT FINALE DEI TEST          ")
        print("==================================================")
        print(json.dumps(self.global_reports, indent=2))
        print("==================================================")

        exec_mode = os.environ.get("RUNNING_IN_DOCKER", "false")
        if self.env == "aws":
            output_dir = "./test_reports/aws"
        elif exec_mode == "true":
            output_dir = "./test_reports/docker"
        else:
            output_dir = "./test_reports/local"
        test_name = "all_tests" if len(self.global_reports) != 1 else next(iter(self.global_reports.keys()))
        suffix = f"_n{self.n_samples_label}" if self.n_samples_label else ""
        output_path = os.path.join(output_dir, f"test_report_{test_name}{suffix}.json")

        try:

            os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.global_reports, f, indent=2)

            print(f"[ENGINE SYSTEM] Report delle metriche salvato in: '{output_path}'")
        except IOError as e:
            print(f"[ENGINE ERRORE] Impossibile salvare il report su disco: {e}")

        if self.env == "aws":
            self._upload_report_to_s3(output_path, test_name, suffix)

    def _upload_report_to_s3(self, local_path: str, test_name: str, suffix: str = ""):
        bucket_name = os.environ.get("DATASETS_BUCKET_NAME")
        if not bucket_name:
            print("[ENGINE] [WARN] DATASETS_BUCKET_NAME non impostata: salto l'upload del report su S3.")
            return
        try:
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            s3_client = boto3.client("s3", region_name=region)
            timestamp = int(time.time())
            s3_key = f"test_reports/aws/test_report_{test_name}{suffix}_{timestamp}.json"
            s3_client.upload_file(local_path, bucket_name, s3_key)
            print(f"[ENGINE SYSTEM] Report caricato anche su S3: s3://{bucket_name}/{s3_key}")
        except Exception as e:
            print(f"[ENGINE] [WARN] Upload del report su S3 fallito (il file resta comunque su disco locale del task): {e}")


if __name__ == "__main__":
    mode = os.environ.get("TRAINING_MODE", "centralized")
    env = os.environ.get("ENV_MODE", "local")
    print(f"[ENGINE] Modalità di addestramento: {mode}, Ambiente: {env}")
    engine = TestEngine(mode=mode, env=env)
    engine.run_scenarios()