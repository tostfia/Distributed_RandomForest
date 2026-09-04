import time
import threading
import os
import signal
from src.testing.scenarios.base import BaseTestScenario


def _merge_aws_overrides(config: dict, key: str) -> dict:
    """
    Unisce il blocco di config 'key' (es. 'fault_tolerance') con l'eventuale
    override AWS-specifico in config['aws']['suggested_overrides'][key] —
    quest'ultimo, se presente, vince sui valori "locali". Pensato per i
    timeout di guasto: su AWS l'ETL/RPC reale è più lento del kill istantaneo
    locale/Docker (l'ETL da solo impiega spesso 30-40s), quindi un
    kill_worker_after_seconds tarato per il locale scatterebbe troppo presto,
    prima ancora che un worker abbia ricevuto un chunk reale — esercitando
    solo "worker sparito prima dell'assegnazione", non "worker morto a metà
    lavoro". Filtra le chiavi di solo commento (es. '_NOTE') presenti nel
    JSON di config.
    """
    merged = dict(config.get(key, {}) or {})
    if (config.get("aws", {}) or {}).get("suggested_overrides", {}).get(key):
        overrides = config["aws"]["suggested_overrides"][key]
        merged.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return merged


def _resolve_aws_infra(config: dict):
    """
    Cluster/region/nomi service ECS: letti da config['aws'] quando presente
    (stessa sezione già usata da run_test_aws.sh/aws_ecs_utils.py, vedi
    test_config.json), altrimenti fallback su env var/default fisso — così
    il comportamento resta invariato anche se il config non ha quella sezione.
    """
    aws_cfg = config.get("aws", {}) or {}
    cluster = aws_cfg.get("ecs_cluster_name") or os.environ.get("CLUSTER_NAME", "forest-cluster")
    region = aws_cfg.get("region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    worker_service_centralized = aws_cfg.get("worker_service_name_centralized", "worker-service")
    worker_service_federated_prefix = aws_cfg.get("worker_service_name_federated_prefix", "worker-service-")
    return cluster, region, worker_service_centralized, worker_service_federated_prefix


def _kill_one_ecs_worker_task(mode: str, config: dict, worker_index: int = 1):
    """
    Simula il crash improvviso di UN worker Fargate fermando fisicamente il suo
    task ECS (ecs:StopTask) — l'equivalente AWS del 'docker kill worker-1' usato
    in locale/Docker Compose. Su Fargate non esiste un docker.sock raggiungibile
    dal task del test-engine, e la porta RPC 18861 del worker non è comunque
    quella del container del test-engine (ogni worker ha la propria ENI/IP): né
    il ramo Docker né quello 'lsof sulla porta locale' possono funzionare qui.

    In centralized il worker scelto è arbitrario (sono intercambiabili per
    design, vedi commento in Terraform, ecs_task_definitions.tf) e worker_index viene ignorato: si
    colpisce sempre 'worker-service'. In federated, worker_index (1-based)
    viene scelto dal chiamante (vedi
    BaseTestScenario._pick_worker_index_with_real_work) invece di essere
    sempre fisso a 1: con l'allocazione proporzionale degli alberi
    (FederatedOrchestrator._allocate_tree_quotas), un worker con shard
    piccolo/vuoto (tipico con partizionamento Dirichlet ad alpha basso)
    potrebbe non avere ricevuto alcun lavoro reale da redistribuire.

    Se il worker-service ha desired-count > 0 (sempre, salvo teardown), ECS
    pianifica automaticamente un task di rimpiazzo: è l'equivalente Fargate del
    supervisor locale che fa restart/backoff del processo worker ucciso.

    Cluster/region/nome service letti da config['aws'] (vedi _resolve_aws_infra),
    con fallback su env var/default fisso se quella sezione manca.
    """
    try:
        import boto3
    except ImportError:
        print("[TEST ERRORE] Pacchetto 'boto3' non disponibile: impossibile fermare un task ECS.")
        return

    cluster, region, worker_service_centralized, worker_service_federated_prefix = _resolve_aws_infra(config)
    service_name = worker_service_centralized if mode == "centralized" else f"{worker_service_federated_prefix}{worker_index}"

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
        ft_cfg = _merge_aws_overrides(self.config, "fault_tolerance")

        mode = os.environ.get("TRAINING_MODE", "centralized")

        # Numero di alberi dal manifesto della baseline (vedi
        # BaseTestScenario._resolve_hyperparameters): stessa fonte del payload,
        # quindi non si puo' piu' chiedere N alberi dichiarandone M ai worker.
        target_trees = self._resolve_target_trees()

        # Segnala al thread killer che il job sotto test è già concluso: senza
        # questo, un guasto programmato DOPO la fine del job (kill_delay più
        # lungo del tempo reale di training) scatterebbe comunque, ma
        # colpendo un job/scenario SUCCESSIVO — esattamente il bug osservato
        # eseguendo 'all' (il kill dello scenario 4 è comparso durante lo
        # scenario 6). Il thread lo controlla PRIMA di procedere al kill vero
        # e proprio, e se il job è già finito annulla il guasto invece di
        # spararlo a vuoto/tardi.
        job_done_event = threading.Event()
        fault_triggered = {"value": False}

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
                    # Aspettiamo il tempo residuo, ma ci fermiamo prima se il
                    # job termina nel frattempo: niente kill_delay fisso che
                    # spara dopo che il job (e magari lo scenario) è già finito.
                    if job_done_event.wait(timeout=remaining):
                        print("[TEST] Il job è già COMPLETATO prima dello scadere del timer di guasto: "
                              "guasto ANNULLATO (altrimenti colpirebbe il prossimo scenario, non questo).")
                        return

            if job_done_event.is_set():
                print("[TEST] Job già completato: guasto ANNULLATO.")
                return

            environment = getattr(self.orchestrator, "environment", "local")
            is_docker = os.environ.get("RUNNING_IN_DOCKER") == "true"
            print("\n[TEST TRIGGER] Simulo guasto imprevisto: Interrompo forzatamente una connessione Worker...")

            # In centralized qualunque worker va bene (sono intercambiabili per
            # design). In federated, invece, "sempre worker 1" rischiava di
            # colpire un worker che con l'allocazione proporzionale degli
            # alberi non ha ricevuto alcun lavoro reale (shard piccolo/vuoto,
            # tipico con Dirichlet ad alpha basso) — vedi
            # BaseTestScenario._pick_worker_index_with_real_work.
            target_worker_index = 1
            if mode == "federated":
                target_worker_index = self._pick_worker_index_with_real_work(environment, default_index=1)

            try:
                if environment == "aws":
                    # RUNNING_IN_DOCKER=true è settato anche su AWS/ECS Fargate,
                    # ma qui non c'è né un docker.sock raggiungibile né un
                    # processo worker locale sulla porta 18861: va sempre presa
                    # la via ECS (ecs:StopTask), indipendentemente da is_docker.
                    _kill_one_ecs_worker_task(mode, self.config, worker_index=target_worker_index)
                elif is_docker:
                    import docker
                    client = docker.from_env()
                    containers = client.containers.list(filters={"label": "com.docker.compose.service=worker"})
                    target_name_fragment = f"worker-{target_worker_index}"
                    for c in containers:
                        if target_name_fragment in c.name:
                            c.kill()
                            break
                else:
                    worker_port = 18861 + target_worker_index - 1
                    cmd_out = os.popen(f"lsof -t -i:{worker_port} 2>/dev/null || fuser {worker_port}/tcp 2>/dev/null").read().strip()
                    if cmd_out:
                        pids = cmd_out.split()
                        my_pid = str(os.getpid())
                        valid_pids = [p for p in pids if p != my_pid]
                        if valid_pids:
                            # Uccidiamo il figlio. Il supervisor capterà l'exit code != 0 e farà il backoff
                            os.kill(int(valid_pids[0]), signal.SIGKILL)
                            print(f"[TEST TRIGGER] Processo Worker locale (PID {valid_pids[0]}) abbattuto!")
                fault_triggered["value"] = True
            except Exception as e:
                    print(f"[TEST ERRORE] Impossibile eseguire il kill: {e}")

        # Reset esplicito: chunk_sent_event potrebbe essere ancora "set" da un
        # job/scenario precedente eseguito nello stesso processo (es. con
        # 'all'). Senza questo clear, il thread killer lo vedrebbe già
        # segnalato e passerebbe direttamente al kill_delay pieno come se il
        # chunk fosse appena partito — o peggio, con un chunk_sent_event già
        # settato e un kill_delay più lungo della durata reale del job,
        # scatterebbe più tardi, durante lo scenario SUCCESSIVO.
        self.orchestrator.chunk_sent_event.clear()

        kill_thread = threading.Thread(target=kill_worker_local, daemon=True)
        kill_thread.start()
        start_time = time.perf_counter()

        payload = self._build_payload()
        self._reuse_dataset_if_available(payload, seed=123)
        num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
        duration = time.perf_counter() - start_time

        # Segnaliamo la fine del job e aspettiamo che il thread killer si
        # concluda (annullandosi o sparando) PRIMA di ritornare: altrimenti,
        # se il job finisce prima del kill_delay, il thread resterebbe vivo
        # (daemon=True, mai altrimenti atteso/cancellato) e potrebbe sparare
        # il kill nel bel mezzo dello scenario successivo — il bug osservato.
        job_done_event.set()
        kill_delay = ft_cfg.get("kill_worker_after_seconds") or 0
        kill_thread.join(timeout=max(kill_delay, 5) + 5)
        if kill_thread.is_alive():
            print("[TEST WARN] Il thread di simulazione del guasto non si è concluso in tempo "
                  "(probabilmente bloccato su una chiamata di rete): potrebbe sparare durante lo "
                  "scenario successivo. Valuta un timeout più basso, o ignora se è l'ultimo scenario eseguito.")

        self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)

        return {
            "scenario_description": "Crash improvviso Worker su thread/processi Python locali.",
            "execution_mode": "centralized" if mode == "centralized" else "federated",
            "status": "SUCCESS" if num_trees == target_trees else "FAILED",
            "trees_built": num_trees,
            "duration_seconds": round(duration, 2),
            # Se False, il guasto è stato annullato (job finito prima del
            # kill_delay) o il thread non ha fatto in tempo a confermarlo:
            # uno "status": "SUCCESS" con questo a False NON ha davvero
            # testato la tolleranza ai guasti, va scartato dal confronto.
            "fault_actually_triggered": fault_triggered["value"],
        }

    def _build_payload(self):
        # Iperparametri dal manifesto della baseline: fonte unica condivisa
        # con run_baseline() (vedi BaseTestScenario._resolve_hyperparameters).
        hp = self._resolve_hyperparameters()
        payload = {
            "job_id": f"test_fault_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": hp,
        }
        # Solo in federato: senza questo, i log/le metriche di questo scenario
        # etichetterebbero sempre "iid"/"proportional" anche se gli shard sul
        # disco sono stati provisionati con Dirichlet/equal (vedi
        # BaseTestScenario._augment_payload_with_partitioning).
        if os.environ.get("TRAINING_MODE", "centralized") == "federated":
            payload = self._augment_payload_with_partitioning(payload)
        return payload