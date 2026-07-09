"""
Implementazione reale dello State Manager su Amazon DynamoDB.

Espone la STESSA interfaccia pubblica di MockStateManager (statemanager.py):
    - initiate_request, obtain_request, update_request_status, complete_request  (astratti)
    - register_worker_task, update_worker_task_status, are_all_workers_done
    - get_active_jobs, get_job_status

In più implementa i metodi di leadership lock che BaseOrchestrator invoca
quando self.environment != "local" (in locale usa invece un file-lock con
fcntl, vedi BaseOrchestrator._try_acquire_leadership):
    - acquire_global_lock(lock_key, owner, ttl=30) -> bool
    - refresh_global_lock(lock_key, owner, ttl=30) -> bool
    - release_global_lock(lock_key, owner) -> bool

A differenza del mock, che importa `dynamo_db` direttamente, questa classe
passa sempre attraverso DynamoDBFactory: così tutto il sistema condivide
la stessa istanza di AwsDynamoDB (e la sua cache di tabelle boto3).
"""

import time
from typing import Optional

from src.shared.mock_aws.dynamodb.dynamodb_factory import DynamoDBFactory
from src.shared.mock_aws.interfaces import StateManagerInterface


JOBS_TABLE = "ModelStatus"
WORKER_TASKS_TABLE = "WorkerTasks"
LOCKS_TABLE = "OrchestratorLocks"


class AwsStateManager(StateManagerInterface):

    def __init__(self, region_name: Optional[str] = None):
        self._db = DynamoDBFactory.get_db("aws", region_name=region_name)

    @staticmethod
    def _unwrap(response: Optional[dict]) -> dict:
        """get_item restituisce {'Item': {...}} oppure {}; qui normalizziamo a dict piatto."""
        if not response:
            return {}
        return response.get("Item", response) if isinstance(response, dict) else {}

    # ------------------------------------------------------------------
    # Ciclo di vita dei job (stessa logica del mock, backend reale)
    # ------------------------------------------------------------------

    def initiate_request(self, job_id: str, dataset_path: str, seed: int) -> None:
        payload = {
            "status": "QUEUED",
            "dataset_path": dataset_path,
            "timestamp": time.time(),
            "retries": 0,
            "last_orchestrator": None,
            "alberi_addestrati": 0,
            "base_random_state": seed,
        }
        self._db.put_item(JOBS_TABLE, job_id, payload)
        print(f"[AWS StateManager] Richiesta registrata (QUEUED) per Job ID: {job_id[:8]}... con Seed: {seed}")

    def obtain_request(self, job_id: str) -> Optional[dict]:
        return self._db.get_item(JOBS_TABLE, job_id)

    def update_request_status(
        self,
        job_id: str,
        status: str,
        orchestrator_id: str,
        retries: int = 0,
        base_random_state: Optional[int] = None,
        alberi_addestrati: int = 0,
    ) -> None:
        current_item = self._unwrap(self.obtain_request(job_id))
        final_seed = base_random_state if base_random_state is not None else current_item.get("base_random_state")

        payload = {
            "status": status,
            "dataset_path": current_item.get("dataset_path"),
            "timestamp": time.time(),
            "retries": retries,
            "last_orchestrator": orchestrator_id,
            "base_random_state": final_seed,
            "alberi_addestrati": alberi_addestrati,
        }
        self._db.put_item(JOBS_TABLE, job_id, payload)

        info_progress = (
            f" | Alberi fatti: {alberi_addestrati} | Seed: {final_seed}"
            if alberi_addestrati > 0
            else f" | Seed: {final_seed}"
        )
        print(f"[AWS StateManager] Job ID: {job_id[:8]}... aggiornato a stato: {status} da {orchestrator_id}{info_progress}")

    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        current_item = self._unwrap(self.obtain_request(job_id))

        payload = {
            "status": "COMPLETED",
            "dataset_path": current_item.get("dataset_path"),
            "timestamp": time.time(),
            "retries": current_item.get("retries", 0),
            "last_orchestrator": orchestrator_id,
            "base_random_state": current_item.get("base_random_state"),
            "alberi_addestrati": current_item.get("alberi_addestrati", 0),
        }
        self._db.put_item(JOBS_TABLE, job_id, payload)
        print(
            f"[AWS StateManager] Job ID: {job_id[:8]}... COMPLETATO con successo da {orchestrator_id} "
            f"| Seed finale: {payload['base_random_state']}"
        )

    # ------------------------------------------------------------------
    # Tracciamento dei worker task
    # ------------------------------------------------------------------

    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        task_id = f"{job_id}#{worker_id}"
        payload = {
            "status": status,
            "timestamp": time.time(),
            "job_id": job_id,
            "worker_id": worker_id,
        }
        self._db.put_item(WORKER_TASKS_TABLE, task_id, payload)
        print(f"[AWS StateManager] Task registrato: Job {job_id[:8]} -> Worker {worker_id} in stato {status}")

    def update_worker_task_status(self, job_id: str, worker_id: str, status: str) -> None:
        task_id = f"{job_id}#{worker_id}"
        current_item = self._unwrap(self._db.get_item(WORKER_TASKS_TABLE, task_id))
        current_item.update({"status": status, "timestamp": time.time()})
        self._db.put_item(WORKER_TASKS_TABLE, task_id, current_item)
        print(f"[AWS StateManager] Worker {worker_id} ha aggiornato status a {status}")

    def are_all_workers_done(self, job_id: str, expected_count: int) -> bool:
        response = self._db.scan_table(WORKER_TASKS_TABLE)
        all_tasks = response.get("Items", [])

        job_tasks = [t for t in all_tasks if t.get("job_id") == job_id]
        completed_tasks = [t for t in job_tasks if t.get("status") == "COMPLETED"]

        print(f"[AWS StateManager] Job {job_id[:8]} -> Worker pronti: {len(completed_tasks)}/{expected_count}")
        return len(completed_tasks) == expected_count

    def get_active_jobs(self) -> list:
        response = self._db.scan_table(JOBS_TABLE)
        all_jobs = response.get("Items", [])
        active_ids = [j.get("job_id") for j in all_jobs if j.get("status") == "PROCESSING" and j.get("job_id")]
        print(f"[AWS StateManager] Scansione job attivi: trovati {len(active_ids)} job in stato PROCESSING.")
        return active_ids

    def get_job_status(self, job_id: str) -> Optional[str]:
        item = self._unwrap(self._db.get_item(JOBS_TABLE, job_id))
        return item.get("status") if item else None

    # ------------------------------------------------------------------
    # Leadership lock globale (solo per ambiente 'aws')
    # ------------------------------------------------------------------

    def acquire_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        return self._db.try_acquire_lock(LOCKS_TABLE, lock_key, owner, ttl)

    def refresh_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        # Stessa scrittura condizionata: si applica solo se siamo ancora leader
        # (o se il lock è scaduto, caso limite che non dovrebbe verificarsi
        # se il refresh avviene con la cadenza attesa dall'heartbeat).
        return self._db.try_acquire_lock(LOCKS_TABLE, lock_key, owner, ttl)

    def release_global_lock(self, lock_key: str, owner: str) -> bool:
        return self._db.release_lock(LOCKS_TABLE, lock_key, owner)