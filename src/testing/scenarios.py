import time
import json
import threading
from abc import ABC, abstractmethod
import subprocess
import time

class BaseTestScenario(ABC):
    """Classe base astratta per tutti gli scenari di test."""
    def __init__(self, config: dict, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

    @abstractmethod
    def run(self) -> dict:
        """Esegue lo scenario e restituisce un dizionario con i risultati/metriche."""
        pass


class PerformanceAndMetricsScenario(BaseTestScenario):
    """Copre lo Scenario 1 e 4: Valutazione Prestazioni (Classif./Regr.) e Analisi Metriche."""
    def run(self) -> dict:
        print(f"\n--- [SCENARIO 1 & 4] Test Prestazioni e Metriche per: {self.config['selected_task']} ---")
        
        payload = {
            "job_id": f"test_perf_{int(time.time())}",
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {
                "n_estimators": 30,
                "max_depth": 5,
                "tree_type": self.config["selected_task"]
            }
        }
        
        start_time = time.perf_counter()
        # Invoca la logica dell'orchestrator passato come dipendenza
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=30, seed=42)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = num_trees / duration if duration > 0 else 0
        
        # Qui il tuo sistema internamente chiamerà _print_and_validate_metrics()
        # Raccogliamo i risultati strutturati
        return {
            "status": "SUCCESS" if num_trees == 30 else "FAILED",
            "duration_seconds": duration,
            "trees_built": num_trees,
            "throughput_trees_per_sec": throughput
        }


class ScalabilityScenario(BaseTestScenario):
    """Copre lo Scenario 2: Analisi della Scalabilità e del Throughput al variare dei Worker."""
    def run(self) -> dict:
        print("\n--- [SCENARIO 2] Test di Scalabilità e Throughput ---")
        scal_cfg = self.config["scalability_test"]
        results = {}
        
        # Eseguiamo il test ciclicamente per i diversi numeri di worker configurati
        for worker_count in scal_cfg["worker_counts_to_test"]:
            print(f"Simulazione/Test con {worker_count} Worker attivi...")
            
            # Nota: In base a come gestisci i worker, qui potresti dover fare il prune 
            # temporaneo del ServiceRegistry o filtrare i canali RPC attivi nell'orchestrator.
            
            payload = {
                "job_id": f"test_scal_{worker_count}_{int(time.time())}",
                "dataset_type": self.config["dataset_type"],
                "dataset_path": self.config["dataset_path"],
                "hyperparameters": {
                    "n_estimators": scal_cfg["n_estimators_per_worker"] * worker_count,
                    "max_depth": 5,
                    "tree_type": self.config["selected_task"]
                }
            }
            
            start_time = time.perf_counter()
            total_target = scal_cfg["n_estimators_per_worker"] * worker_count
            num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=total_target, seed=42)
            duration = time.perf_counter() - start_time
            
            throughput = num_trees / duration if duration > 0 else 0
            results[f"worker_count_{worker_count}"] = {
                "duration": duration,
                "throughput": throughput,
                "trees": num_trees
            }
            print(f"-> Worker: {worker_count} | Tempo: {duration:.2f}s | Throughput: {throughput:.2f} alberi/s")
            
        return results




class NetworkSimulationScenario(BaseTestScenario):
    """Copre lo Scenario 3: Simulazione Reale di Ritardi di Rete tramite tc (Traffic Control)."""
    
    def _apply_tc_rules(self, latency_ms: int, loss_percentage: float):
        """Applica le regole di traffic control sull'interfaccia lo usando sudo."""
        print(f"\n[SYSTEM tc] Configurazione regole di rete su 'lo': +{latency_ms}ms, {loss_percentage}% loss...")
        # Rimuove eventuali regole precedenti per evitare conflitti
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "lo", "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Costruisce il comando netem in base ai parametri del JSON
        cmd = ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "delay", f"{latency_ms}ms"]
        if loss_percentage > 0:
            cmd.extend(["loss", f"{loss_percentage}%"])
            
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"[SYSTEM tc WARNING] Errore nell'applicazione di tc (Richiede privilegi sudo senza password?): {result.stderr.strip()}")

    def _clear_tc_rules(self):
        """Ripristina lo stato pulito dell'interfaccia di rete di loopback."""
        print("[SYSTEM tc CLEANUP] Rimozione dei ritardi di rete da localhost (lo)...")
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "lo", "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self) -> dict:
        net_cfg = self.config.get("network_simulation", {})
        
        # Converte i secondi del JSON in millisecondi (es: 1.5s -> 1500ms)
        latency_ms = int(net_cfg.get("latency_seconds", 0.0) * 1000)
        loss_percentage = float(net_cfg.get("packet_loss_rate", 0.0) * 100) # Se espresso in decimale (0.0 - 1.0)
        
        print(f"\n--- [SCENARIO 3] Simulazione di Rete Reale (tc netem) ---")
        
        # 1. Applica le restrizioni reali sul kernel Linux
        if latency_ms > 0 or loss_percentage > 0:
            self._apply_tc_rules(latency_ms, loss_percentage)
        
        try:
            payload = {
                "job_id": f"test_network_tc_{int(time.time())}",
                "dataset_type": "synthetic",
                "dataset_path": self.config["dataset_path"],
                "hyperparameters": {
                    "n_estimators": 10, 
                    "max_depth": 5, 
                    "tree_type": self.config["selected_task"]
                }
            }
            
            start_time = time.perf_counter()
            # Esegue il training reale: ogni chiamata RPC adesso subirà il lag di tc!
            num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=10, seed=42)
            duration = time.perf_counter() - start_time
            
            return {
                "status": "SUCCESS" if num_trees == 10 else "PARTIAL",
                "duration_with_real_latency_seconds": duration,
                "trees_built": num_trees,
                "applied_latency_ms": latency_ms,
                "applied_packet_loss_percent": loss_percentage
            }
            
        finally:
            # 2. Il blocco finally garantisce il ripristino della rete lo anche se il test fallisce o crasha
            self._clear_tc_rules()

class FaultToleranceScenario(BaseTestScenario):
    """Copre lo Scenario 5: Sperimentazione della Tolleranza ai Guasti (Failover)."""
    def run(self) -> dict:
        ft_cfg = self.config["fault_tolerance"]
        print("\n--- [SCENARIO 5] Sperimentazione della Tolleranza ai Guasti (Kill Worker) ---")
        
        def kill_worker_target():
            time.sleep(ft_cfg["kill_worker_after_seconds"])
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker...")
            # Logica per simulare il crash di un worker. 
            # Esempio: chiudere forzatamente una delle connessioni RPyC nell'orchestratore
            if hasattr(self.orchestrator, "worker_channels") and self.orchestrator.worker_channels:
                try:
                    target_worker = list(self.orchestrator.worker_channels.keys())[0]
                    self.orchestrator.worker_channels[target_worker].close()
                    print(f"[TEST TRIGGER] Connessione con il Worker {target_worker} interrotta.")
                except Exception as e:
                    print(f"[TEST TRIGGER ERRORE] Impossibile chiudere il worker: {e}")

        # Avvia il thread killer in background che agirà durante l'addestramento
        killer_thread = threading.Thread(target=kill_worker_target)
        killer_thread.daemon = True
        
        payload = {
            "job_id": f"test_fault_{int(time.time())}",
            "dataset_type": self.config["dataset_type"],
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {"n_estimators": 60, "max_depth": 5, "tree_type": self.config["selected_task"]}
        }
        
        killer_thread.start()
        
        start_time = time.perf_counter()
        # _execute_training_step intercetterà l'eccezione RPC del worker chiuso, 
        # reinserirà il chunk nella coda e terminerà l'addestramento grazie agli altri worker.
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=60, seed=42)
        duration = time.perf_counter() - start_time
        
        status = "SUCCESS" if num_trees >= ft_cfg["expected_min_trees"] else "FAILED"
        return {
            "status": status,
            "duration_seconds": duration,
            "trees_built": num_trees
        }