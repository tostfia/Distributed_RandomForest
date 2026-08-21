import time
import threading
import os
import signal
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
    il job: qualunque worker fermato esercita comunque il path di fault-tolerance
    (redistribuzione del chunk sui worker superstiti).

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

        # Scelta deterministica (ordine ARN) per riproducibilità tra run.
        target_arn = sorted(task_arns)[0]
        ecs.stop_task(
            cluster=cluster,
            task=target_arn,
            reason="[TEST] Simulazione crash worker (fault tolerance scenario)"
        )
        print(f"[TEST TRIGGER] Task Fargate del worker fermato: {target_arn.split('/')[-1]} "
              f"(service '{service_name}', cluster '{cluster}').")
    except Exception as e:
        print(f"[TEST ERRORE] Impossibile fermare il task ECS del worker: {e}")


class FaultToleranceScenario(BaseTestScenario):
    """Copre lo Scenario 5: Sperimentazione della Tolleranza ai Guasti (Kill Worker)."""


    def run(self) -> dict:
        ft_cfg = self.config.get("fault_tolerance", {})

        mode = os.environ.get("SYS_MODE", "centralized")

        task_type = self.config.get("selected_task", "classifier")
        if task_type == "classifier":
            target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
        else:
            target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)
        def kill_worker_local():
            # Rispettiamo entrambi gli intenti: il guasto non scatta MAI prima che
            # sia stato inviato almeno un chunk reale (altrimenti staremmo testando
            # il recupero di zero lavoro fatto), ma nemmeno prima che siano passati
            # almeno kill_delay secondi (il config vuole dare modo a un po' di
            # lavoro reale di essere completato prima di simulare il crash).
            kill_delay = ft_cfg.get("kill_worker_after_seconds")

            wait_start = time.perf_counter()
            signaled = self.orchestrator.chunk_sent_event.wait(timeout=kill_delay)
            if not signaled:
                print(f"[TEST WARN] Timeout di {kill_delay} secondi raggiunto senza che il chunk sia stato inviato. Procedo comunque a simulare il guasto.")
            else:
                elapsed = time.perf_counter() - wait_start
                remaining = kill_delay - elapsed
                if remaining > 0:
                    print(f"[TEST] Primo chunk inviato dopo {elapsed:.1f}s. Attendo altri {remaining:.1f}s "
                          f"(fino a {kill_delay}s totali) per dare modo a un po' di lavoro reale di completarsi...")
                    time.sleep(remaining)

            environment = getattr(self.orchestrator, "environment", "local")
            is_docker = os.environ.get("RUNNING_IN_DOCKER") == "true"
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


        threading.Thread(target=kill_worker_local, daemon=True).start()
        start_time = time.perf_counter()

        payload = self._build_payload()
        self._reuse_dataset_if_available(payload, seed=123)
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
        duration = time.perf_counter() - start_time
        self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)

        return {
            "scenario_description": "Crash improvviso Worker su thread/processi Python locali.",
            "execution_mode": "centralized" if mode == "centralized" else "federated",
            "status": "SUCCESS" if num_trees == target_trees else "FAILED",
            "trees_built": num_trees,
            "duration_seconds": round(duration, 2)
        }

    def _build_payload(self):
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        return {
            "job_id": f"test_fault_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": hp,
        }