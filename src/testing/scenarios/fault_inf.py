import signal
import time
import threading
import os
from src.testing.scenarios.base import BaseTestScenario


def _kill_one_ecs_worker_task(mode: str):
    """
    Simula il crash improvviso di UN worker Fargate fermando fisicamente il suo
    task ECS (ecs:StopTask) — l'equivalente AWS del 'docker kill worker-1' usato
    in locale/Docker Compose. Su Fargate non esiste un docker.sock raggiungibile
    dal task del test-engine, e la porta RPC 18861 del worker non è comunque
    quella del container del test-engine (ogni worker ha la propria ENI/IP): né
    il ramo Docker né quello 'lsof sulla porta locale' possono funzionare qui.

    Colpisce sempre il "worker 1" (worker-service in centralized, worker-service-1
    in federated), analogamente a come il ramo Docker punta sempre al container
    'worker-1' — non serve individuare quale worker specifico stia processando
    l'inferenza: qualunque worker fermato esercita comunque il path di
    fault-tolerance (redistribuzione del chunk sui worker superstiti).

    Se il worker-service ha desired-count > 0 (sempre, salvo teardown), ECS
    pianifica automaticamente un task di rimpiazzo: è l'equivalente Fargate del
    supervisor locale che fa restart/backoff del processo worker ucciso.

    CLUSTER_NAME non è oggi passato come env var ai task (deploy.sh lo tiene
    solo come variabile bash): usiamo lo stesso pattern di fallback già in uso
    altrove nel progetto per BUCKET_NAME, con default 'forest-cluster' (il nome
    fisso usato da deploy.sh/teardown.sh).
    """
    try:
        import boto3
    except ImportError:
        print("[TEST ERRORE] Pacchetto 'boto3' non disponibile: impossibile fermare un task ECS.")
        return

    cluster = os.environ.get("CLUSTER_NAME", "forest-cluster")
    service_name = "worker-service" if mode == "centralized" else "worker-service-1"
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    try:
        ecs = boto3.client("ecs", region_name=region)
        task_arns = ecs.list_tasks(
            cluster=cluster, serviceName=service_name, desiredStatus="RUNNING"
        ).get("taskArns", [])

        if not task_arns:
            print(f"[TEST ERRORE] Nessun task RUNNING trovato per il service '{service_name}' "
                  f"sul cluster '{cluster}'. Impossibile simulare il crash.")
            return

        target_arn = sorted(task_arns)[0]
        ecs.stop_task(
            cluster=cluster,
            task=target_arn,
            reason="[TEST] Simulazione crash worker durante inferenza (fault tolerance scenario)"
        )
        print(f"[TEST TRIGGER] Task Fargate del worker fermato: {target_arn.split('/')[-1]} "
              f"(service '{service_name}', cluster '{cluster}').")
    except Exception as e:
        print(f"[TEST ERRORE] Impossibile fermare il task ECS del worker: {e}")


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

            environment = getattr(self.orchestrator, "environment", "local")
            is_docker = os.environ.get("RUNNING_IN_DOCKER") == "true"
            mode = os.environ.get("SYS_MODE", "centralized")
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker...")
            try:
                if environment == "aws":
                    # RUNNING_IN_DOCKER=true è settato anche su AWS/ECS Fargate,
                    # ma qui non c'è né un docker.sock raggiungibile né un
                    # processo worker locale sulla porta 18861: va sempre presa
                    # la via ECS (ecs:StopTask), indipendentemente da is_docker.
                    _kill_one_ecs_worker_task(mode)
                elif is_docker:
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