import signal
import time
import threading
import os
from src.testing.scenarios.base import BaseTestScenario

class InferenceWorkerFaultScenario(BaseTestScenario):
    """ Copre lo Scenario: Failover del Worker durante la fase di inferenza."""

    def run(self) -> dict:
        ft_cfg = self.config.get("inference_worker_fault", {})
        kill_delay = ft_cfg.get("kill_worker_after_seconds")
        task_type = self.config.get("selected_task", "classifier")
        if task_type == "classifier":
            target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
        else:
            target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)
        payload = self._build_payload()

        print(f"\n[TEST] Caricamento/Generazione modello preliminare per Job: {payload['job_id']}...")
        try:
            # Generiamo pochi alberi (es. 10 o 20) solo per creare legalmente il file .pkl su disco
            # Eseguiamo questa fase senza killare nessuno, in totale stabilità
            self._reuse_dataset_if_available(payload, seed=123)
            num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
            self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)
            print("[TEST] Modello globale generato con successo su disco. Pronto per il test di inferenza.")
        except Exception as e:
            print(f"[TEST ERRORE] Impossibile completare l'addestramento preliminare: {e}")
            return {
                "scenario_description": "Crash improvviso Worker durante l'inferenza (Fallito in setup).",
                "status": "FAILED",
                "error": f"Fase di addestramento preliminare fallita: {e}"
            }
        
        def kill_worker_local():
            signaled = self.orchestrator.chunk_sent_event.wait(timeout=kill_delay)
            if not signaled:
                print(f"[TEST WARN] Timeout di {kill_delay} secondi raggiunto senza che il chunk sia stato inviato. Procedo comunque a simulare il guasto.")
            
            is_docker = os.environ.get("RUNNING_IN_DOCKER") == "true"
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker (Locale)...")
            try:
                if is_docker:
                    import docker
                    client = docker.from_env()
                    containers = client.containers.list(filters={"label": "com.docker.compose.service=worker"})
                    for c in containers:
                        if "worker-1" in c.name:
                            c.kill()
                            break
                else:
                    worker_port = 18861
                    cmd_out = os.popen(f"lsof -t -i:{worker_port} 2>/dev/null || fuser {worker_port}/tcp 2>/dev/null").read().strip()
                    if cmd_out:
                        pids = cmd_out.split()
                        my_pid = str(os.getpid())
                        valid_pids = [p for p in pids if p != my_pid]
                        if valid_pids:
                            # Uccidiamo il figlio. Il supervisor capterà l'exit code != 0 e farà il backoff
                            os.kill(int(valid_pids[0]), signal.SIGKILL)
                            print(f"[TEST TRIGGER] Processo Worker locale (PID {valid_pids[0]}) abbattuto!")
            except Exception as e:
                    print(f"[TEST ERRORE] Impossibile eseguire il kill: {e}")

                # Reset esplicito: chunk_sent_event potrebbe essere ancora "set" dall'ultimo
        # chunk inviato durante il training preliminare qui sopra. Senza questo clear,
        # il thread di kill lo vedrebbe già segnalato e scatterebbe subito, prima che
        # un worker riceva un vero task di inferenza.
        self.orchestrator.chunk_sent_event.clear()

        threading.Thread(target=kill_worker_local, daemon=True).start()
        start_time = time.perf_counter()
        
        test_status = "FAILED"
        accuracy_metrics = None
        
        try:
            print("[TEST] Avvio dell'inferenza distribuita federata (il modello esiste, ora simulo il guasto)...")
            # Adesso l'orchestratore troverà il file .pkl e inizierà a inviare i chunk di test ai worker
            result = self.orchestrator._execute_inference_step(payload) or {}
            accuracy_metrics = result.get("metrics", {})
            
            # Se l'orchestratore gestisce l'eccezione di rete del worker deceduto redistribuendo i chunk,
            # arriverà a fine metodo restituendo le metriche corrette.
            test_status = "SUCCESS"
        except Exception as e:
            print(f"[TEST FAILED] L'orchestratore non ha tollerato il crash del worker in inferenza: {e}")
            test_status = "FAILED"
            
        duration = time.perf_counter() - start_time
        mode = os.environ.get("SYS_MODE", "centralized")
        return {
            "scenario_description": "Crash improvviso Worker su thread/processi Python locali durante l'inferenza.",
            "execution_mode": "centralized" if mode == "centralized" else "federated",
            "status": test_status,
            "duration_seconds": round(duration, 2),
            "accuracy_metrics": accuracy_metrics
        }
    
    def _build_payload(self):
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        
        return {
            "job_id": f"test_inference_fault_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": hp,
        }