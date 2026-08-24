"""
Helper condivisi per gli scenari di test eseguiti contro l'infrastruttura
AWS ECS/Fargate (deploy.sh: cluster 'forest-cluster', service
'orchestrator-service' desired-count=2, service 'worker-service' o
'worker-service-1'..'worker-service-N').

Questo modulo NON tocca il codice degli scenari 'locale/docker' esistenti:
viene importato solo dai nuovi rami 'if aws_env:' aggiunti a fault.py,
fault_inf.py, orchestrator_fault.py, orchestrator_fault_inf.py.

Principio di funzionamento per identificare il LEADER reale:
- Il lock di leadership (tabella DynamoDB 'OrchestratorLocks', stessa
  tabella usata in locale) contiene un campo 'leader' con il nome interno
  dell'orchestratore, che include l'hostname del container (vedi
  _resolve_leader_container in orchestrator_fault.py per il caso Docker
  locale). Su Fargate, anche se non c'è un daemon Docker raggiungibile da
  fuori, ECS espone comunque l'ID del container Docker sottostante nel
  campo 'runtimeId' di 'ecs describe-tasks' (containers[].runtimeId) — è
  lo stesso ID a 12+ caratteri esadecimali che Docker userebbe come
  hostname di default. Il fragment a 12 esadecimali estratto dal nome del
  leader viene quindi confrontato con il prefisso di runtimeId dei task
  RUNNING del service 'orchestrator-service' per determinare quale dei
  due Task ARN è il vero leader.
"""

import os
import re
import boto3

CLUSTER_NAME = os.environ.get("ECS_CLUSTER_NAME", "forest-cluster")
ORCHESTRATOR_SERVICE_NAME = "orchestrator-service"
LOCKS_TABLE = "OrchestratorLocks"

HEX12_RE = re.compile(r"[0-9a-f]{12,}")


def is_aws_environment(orchestrator) -> bool:
    """
    True se lo scenario deve usare i meccanismi di guasto ECS invece di
    quelli Docker-locale/thread-locale.
    """
    env_attr = getattr(orchestrator, "environment", "") or ""
    return env_attr == "aws" or os.environ.get("ENV_MODE", "").lower() == "aws"


def get_ecs_client(region: str = None):
    if boto3 is None:
        raise RuntimeError("boto3 non installato: impossibile usare i meccanismi di guasto AWS/ECS.")
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("ecs", region_name=region)


def _unwrap_dynamo_item(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict) and "Item" in raw:
        return raw.get("Item") or {}
    return raw if isinstance(raw, dict) else {}


def read_orchestrator_lock(state_manager, lock_key: str) -> dict:
    """
    Legge il contenuto grezzo del lock di leadership da DynamoDB
    (tabella OrchestratorLocks), riusando la connessione già aperta
    dallo state_manager AWS (AwsStateManager espone un attributo privato
    '_db' — vedi src/master/orchestrator/.../awsstatemanager.py — con lo
    stesso get_item(table, key) usato per la tabella ModelStatus).

    Ritorna {} se il lock non esiste o in caso di errore (log a video,
    nessuna eccezione propagata: il chiamante gestisce il fallback).
    """
    db = getattr(state_manager, "_db", None)
    if db is None:
        print("[AWS ECS UTILS] [WARN] state_manager privo di attributo '_db': impossibile leggere il lock.")
        return {}
    try:
        raw = db.get_item(LOCKS_TABLE, lock_key)
        return _unwrap_dynamo_item(raw)
    except Exception as e:
        print(f"[AWS ECS UTILS] [WARN] Lettura lock '{lock_key}' fallita: {e}")
        return {}


def list_running_tasks(ecs_client, service_name: str, cluster: str = CLUSTER_NAME) -> list:
    """Ritorna la lista di Task ARN RUNNING per il service indicato."""
    resp = ecs_client.list_tasks(cluster=cluster, serviceName=service_name, desiredStatus="RUNNING")
    return resp.get("taskArns", [])


def describe_tasks(ecs_client, task_arns: list, cluster: str = CLUSTER_NAME) -> list:
    if not task_arns:
        return []
    resp = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns)
    return resp.get("tasks", [])


def resolve_leader_task_arn(ecs_client, state_manager, lock_key: str, cluster: str = CLUSTER_NAME):
    """
    Determina QUALE dei task RUNNING di 'orchestrator-service' sta
    detenendo la leadership in questo momento, leggendo il lock
    condiviso su DynamoDB — equivalente ECS di _resolve_leader_container
    (che in locale legge lo stesso lock da un file JSON / dal client
    Docker). Ritorna il Task ARN del leader, o None se non determinabile
    (lock assente/corrotto, o nessun container corrispondente trovato:
    in questo caso il chiamante deve trattare l'esito come incerto,
    esattamente come fa già il fallback nel ramo Docker locale).
    """
    lock_item = read_orchestrator_lock(state_manager, lock_key)
    leader_name = lock_item.get("leader", "") or ""
    match = HEX12_RE.search(leader_name.lower())
    if not match:
        print(f"[AWS ECS UTILS] [WARN] Nessun fragment esadecimale trovato nel nome leader '{leader_name}'.")
        return None
    hostname_fragment = match.group(0)[:12]

    task_arns = list_running_tasks(ecs_client, ORCHESTRATOR_SERVICE_NAME, cluster)
    tasks = describe_tasks(ecs_client, task_arns, cluster)
    for task in tasks:
        for container in task.get("containers", []):
            runtime_id = (container.get("runtimeId") or "").lower()
            if runtime_id.startswith(hostname_fragment):
                return task.get("taskArn")
    return None


def stop_task(ecs_client, task_arn: str, reason: str, cluster: str = CLUSTER_NAME) -> bool:
    if not task_arn:
        return False
    try:
        ecs_client.stop_task(cluster=cluster, task=task_arn, reason=reason[:255])
        return True
    except Exception as e:
        print(f"[AWS ECS UTILS] [ERROR] stop_task fallito su '{task_arn}': {e}")
        return False


def resolve_worker_service_name(worker_index: int = 1) -> str:
    """
    In modalità federated ogni worker ha un service ECS dedicato a indice
    fisso ('worker-service-1'..'worker-service-N', desired-count=1
    ciascuno, vedi deploy.sh). In modalità centralized esiste un unico
    'worker-service' con worker anonimi/intercambiabili: l'indice non ha
    alcun significato e viene ignorato.
    """
    training_mode = os.environ.get("TRAINING_MODE") or "centralized"
    if training_mode == "federated":
        return f"worker-service-{worker_index}"
    return "worker-service"


def pick_and_kill_worker_task(ecs_client, reason: str, worker_index: int = 1, cluster: str = CLUSTER_NAME):
    """
    Sceglie un task RUNNING del worker-service target e lo termina con
    ecs.stop_task — equivalente ECS del SIGKILL sul PID del worker in
    ascolto sulla porta 18861 usato nei rami locali di fault.py/fault_inf.py.

    In centralized il worker scelto è arbitrario (sono intercambiabili per
    design, vedi commento in deploy.sh), in federated si colpisce sempre
    'worker-service-1' per coerenza col comportamento locale (che colpisce
    sempre lo stesso worker fisso sulla porta 18861).

    Ritorna (task_arn_ucciso, service_name) oppure (None, service_name) se
    non è stato trovato nessun task RUNNING.
    """
    service_name = resolve_worker_service_name(worker_index)
    task_arns = list_running_tasks(ecs_client, service_name, cluster)
    if not task_arns:
        print(f"[AWS ECS UTILS] [ERROR] Nessun task RUNNING trovato per il service '{service_name}'.")
        return None, service_name

    target_arn = task_arns[0]
    ok = stop_task(ecs_client, target_arn, reason, cluster)
    return (target_arn if ok else None), service_name