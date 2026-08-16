import threading
import time
from typing import Optional
from src.shared.mock_aws.dynamodb.dynamodb_mock import dynamo_db
from src.shared.mock_aws.interfaces import StateManagerInterface

TABLE_NAME = "ModelStatus"
JOB_LOCKS_TABLE = "JobLocks"

class MockStateManager(StateManagerInterface):
    _claim_lock = threading.Lock()  # Lock per la gestione della concorrenza nella simulazione
    
    def initiate_request(
        self,
        job_id: str,
        dataset_path: str,
        seed: int,
        hyperparameters: Optional[dict] = None,
        mode: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> None:
        payload = {
            "status": "QUEUED",
            "dataset_path": dataset_path,
            "timestamp": time.time(),
            "retries": 0,
            "last_orchestrator": None,
            "alberi_addestrati": 0,
            "base_random_state": seed,
            "hyperparameters": hyperparameters or {},
            "mode": mode,
            "dataset_type": dataset_type,
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        print(f"[StateManager] Richiesta registrata (QUEUED) per Job ID: {job_id[:8]}... con Seed: {seed}")

    def obtain_request(self, job_id: str) -> Optional[dict]:
        return dynamo_db.get_item(TABLE_NAME, job_id)

    def update_request_status(
        self, 
        job_id: str, 
        status: str, 
        orchestrator_id: str, 
        retries: int = 0, 
        base_random_state: Optional[int] = None,
        alberi_addestrati: int = 0
    ) -> None:
        current_job = self.obtain_request(job_id) or {}
        final_seed = base_random_state if base_random_state is not None else current_job.get("base_random_state")
        
        payload = {
            "status": status,
            "dataset_path": current_job.get("dataset_path"),
            "timestamp": time.time(),
            "retries": retries,
            "last_orchestrator": orchestrator_id,
            "base_random_state": final_seed,   
            "alberi_addestrati": alberi_addestrati,
            # Riportati dal record esistente: put_item sovrascrive l'intero item,
            # quindi senza questo passaggio esplicito questi campi andrebbero persi
            # al primo aggiornamento di stato dopo la creazione del job.
            "hyperparameters": current_job.get("hyperparameters", {}),
            "mode": current_job.get("mode"),
            "dataset_type": current_job.get("dataset_type"),
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        
        info_progress = f" | Alberi fatti: {alberi_addestrati} | Seed: {final_seed}" if alberi_addestrati > 0 else f" | Seed: {final_seed}"
        print(f"[StateManager] Job ID: {job_id[:8]}... aggiornato a stato: {status} da {orchestrator_id}{info_progress}")

    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        current_job = self.obtain_request(job_id) or {}
        payload = {
            "status": "COMPLETED",
            "dataset_path": current_job.get("dataset_path"),
            "timestamp": time.time(),
            "retries": current_job.get("retries", 0),
            "last_orchestrator": orchestrator_id,
            "base_random_state": current_job.get("base_random_state"),
            "alberi_addestrati": current_job.get("alberi_addestrati", 0),
            "hyperparameters": current_job.get("hyperparameters", {}),
            "mode": current_job.get("mode"),
            "dataset_type": current_job.get("dataset_type"),
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        print(f"[StateManager] Job ID: {job_id[:8]}... COMPLETATO con successo da {orchestrator_id} | Seed finale: {payload['base_random_state']}")

    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        task_id = f"{job_id}#{worker_id}"
        payload = {
            "status": status,
            "timestamp": time.time(),
            "job_id": job_id,
            "worker_id": worker_id
        }
        dynamo_db.put_item("WorkerTasks", task_id, payload)
        print(f"[StateManager] Task registrato: Job {job_id[:8]} -> Worker {worker_id} in stato {status}")

    def update_worker_task_status(self, job_id: str, worker_id: str, status: str) -> None:
        task_id = f"{job_id}#{worker_id}"
        current = dynamo_db.get_item("WorkerTasks", task_id) or {}
        current.update({"status": status, "timestamp": time.time()})
        dynamo_db.put_item("WorkerTasks", task_id, current)
        print(f"[StateManager] Worker {worker_id} ha aggiornato status a {status}")

    def are_all_workers_done(self, job_id: str, expected_count: int) -> bool:
        response = dynamo_db.scan_table("WorkerTasks") 
        all_tasks = response.get("Items", [])
        job_tasks = [t for t in all_tasks if t.get('job_id') == job_id]
        completed_tasks = [t for t in job_tasks if t.get('status') == 'COMPLETED']
        
        print(f"[StateManager] Job {job_id[:8]} -> Worker pronti: {len(completed_tasks)}/{expected_count}")
        return len(completed_tasks) == expected_count
    
    def get_active_jobs(self) -> list:
        response = dynamo_db.scan_table(TABLE_NAME)
        all_jobs = response.get("Items", [])
        active_ids = [j.get("job_id") for j in all_jobs if j.get("status") == "PROCESSING" and j.get("job_id")]
        print(f"[StateManager] Scansione job attivi: trovati {len(active_ids)} job in stato PROCESSING.")
        return active_ids

    def get_job_status(self, job_id: str) -> Optional[str]:
        response = dynamo_db.get_item(TABLE_NAME, job_id)
        item = response.get("Item") if isinstance(response, dict) and "Item" in response else response
        if item and isinstance(item, dict):
            return item.get("status")
        return None

    def get_job_details(self, job_id: str) -> Optional[dict]:
        response = dynamo_db.get_item(TABLE_NAME, job_id)
        item = response.get("Item") if isinstance(response, dict) and "Item" in response else response
        if item and isinstance(item, dict):
            return item
        return None

    # In MockStateManager:
    def acquire_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        return dynamo_db.try_acquire_lock("OrchestratorLocks", lock_key, owner, ttl)

    def refresh_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        return dynamo_db.refresh_lock("OrchestratorLocks", lock_key, owner, ttl)

    def release_global_lock(self, lock_key: str, owner: str) -> bool:
        return dynamo_db.release_lock("OrchestratorLocks", lock_key, owner)
    
    def try_claim_job(self, job_id: str, orchestrator_id: str, lease_seconds: int = 300) -> bool:
        """
        Reclama (o rinnova) il possesso esclusivo del job. Ritorna True solo se
        nessun altro Orchestrator ha una lease valida in corso su questo job_id.
        Da chiamare prima di iniziare a processare E periodicamente durante
        l'elaborazione, per rinnovare la lease.
        """
        claimed = dynamo_db.try_acquire_lock(JOB_LOCKS_TABLE, job_id, orchestrator_id, ttl=lease_seconds)
        if not claimed:
            # Se il lock esiste ma è già nostro, try_acquire_lock fallisce
            # (perché non è scaduto): il rinnovo passa da refresh_lock.
            claimed = dynamo_db.refresh_lock(JOB_LOCKS_TABLE, job_id, orchestrator_id, ttl=lease_seconds)
 
        if not claimed:
            print(f"[StateManager] [CLAIM FAILED] Job {job_id[:8]}... già posseduto da un altro Orchestrator.")
        return claimed
    def release_job_lease(self, job_id: str, orchestrator_id: str) -> bool:
        """Rilascia volontariamente la lease (job completato o fallito in modo pulito)."""
        return dynamo_db.release_lock(JOB_LOCKS_TABLE, job_id, orchestrator_id)