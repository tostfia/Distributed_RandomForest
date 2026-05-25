import time
from typing import Optional
from src.shared.mock_aws.dynamodb import dynamo_db

TABLE_NAME = "ModelStatus"

def initiate_request(job_id: str, dataset_path: str) -> None:
    """Registra il job appena nato sul database in stato QUEUED."""
    payload = {
        "status": "QUEUED",
        "dataset_path": dataset_path,
        "timestamp": time.time(),
        "retries": 0,
        "last_orchestrator": None
    }
    dynamo_db.put_item(TABLE_NAME, job_id, payload)
    print(f"[StateManager] Richiesta registrata (QUEUED) per Job ID: {job_id[:8]}...")

def obtain_request(job_id: str) -> Optional[dict]:
    """Ottieni lo stato attuale del job da DynamoDB."""
    return dynamo_db.get_item(TABLE_NAME, job_id)

def update_request_status(job_id: str, status: str, orchestrator_id: str, retries: int = 0) -> None:
    """Aggiorna lo stato del job (es: in PROCESSING) tracciando chi lo sta lavorando."""
    current_job = obtain_request(job_id) or {}
    
    payload = {
        "status": status,
        "dataset_path": current_job.get("dataset_path"),
        "timestamp": time.time(),
        "retries": retries,
        "last_orchestrator": orchestrator_id
    }
    dynamo_db.put_item(TABLE_NAME, job_id, payload)
    print(f"[StateManager] Job ID: {job_id[:8]}... aggiornato a stato: {status} da {orchestrator_id}")

def complete_request(job_id: str, orchestrator_id: str) -> None:
    """Finalizza la richiesta impostando lo stato su COMPLETED."""
    current_job = obtain_request(job_id) or {}
    
    payload = {
        "status": "COMPLETED",
        "dataset_path": current_job.get("dataset_path"),
        "timestamp": time.time(),
        "retries": current_job.get("retries", 0),
        "last_orchestrator": orchestrator_id
    }
    dynamo_db.put_item(TABLE_NAME, job_id, payload)
    print(f"[StateManager] Job ID: {job_id[:8]}... COMPLETATO con successo da {orchestrator_id}")