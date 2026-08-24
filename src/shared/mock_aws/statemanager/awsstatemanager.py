import time
from typing import Optional
from src.shared.mock_aws.interfaces import StateManagerInterface
from src.shared.mock_aws.dynamodb.dynamodb_factory import DynamoDBFactory
from src.shared.config import SystemConfig

JOBS_TABLE = "ModelStatus"
WORKER_TASKS_TABLE = "WorkerTasks"
LOCKS_TABLE = "OrchestratorLocks"
JOB_LOCKS_TABLE = "JobLocks"
# Nome del Global Secondary Index su WorkerTasks che permette di recuperare
# tutti i task di un job senza fare la scansione dell'intera tabella.
WORKER_TASKS_JOB_INDEX = "job_id-index"

cfg = SystemConfig()

class AwsStateManager(StateManagerInterface):

    def __init__(self):
        self._db = DynamoDBFactory.get_db(cfg.env)

    @staticmethod
    def _unwrap(response: Optional[dict]) -> dict:
        if not response:
            return {}
        return response.get("Item", response) if isinstance(response, dict) else {}

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
            "timestamp": int(time.time()),
            "retries": 0,
            "last_orchestrator": None,
            "alberi_addestrati": 0,
            "base_random_state": seed,
            "hyperparameters": hyperparameters or {},
            "mode": mode,
            "dataset_type": dataset_type,
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
            "timestamp": int(time.time()),
            "retries": retries,
            "last_orchestrator": orchestrator_id,
            "base_random_state": final_seed,
            "alberi_addestrati": alberi_addestrati,
            "hyperparameters": current_item.get("hyperparameters", {}),
            "mode": current_item.get("mode"),
            "dataset_type": current_item.get("dataset_type"),
        }
        self._db.put_item(JOBS_TABLE, job_id, payload)
        print(f"[AWS StateManager] Job ID: {job_id[:8]}... aggiornato a stato: {status} da {orchestrator_id}")

    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        current_item = self._unwrap(self.obtain_request(job_id))
        payload = {
            "status": "COMPLETED",
            "dataset_path": current_item.get("dataset_path"),
            "timestamp": int(time.time()),
            "retries": current_item.get("retries", 0),
            "last_orchestrator": orchestrator_id,
            "base_random_state": current_item.get("base_random_state"),
            "alberi_addestrati": current_item.get("alberi_addestrati", 0),
            "hyperparameters": current_item.get("hyperparameters", {}),
            "mode": current_item.get("mode"),
            "dataset_type": current_item.get("dataset_type"),
        }
        self._db.put_item(JOBS_TABLE, job_id, payload)
        print(f"[AWS StateManager] Job ID: {job_id[:8]}... COMPLETATO con successo da {orchestrator_id}")

    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        task_id = f"{job_id}#{worker_id}"
        payload = {
            "status": status,
            "timestamp": int(time.time()),
            "job_id": job_id,
            # NOTA: MockStateManager scrive lo stesso dato sotto la chiave
            # 'worker_id'. La differenza è storica e oggi innocua (nessun
            # consumatore legge questo campo: are_all_workers_done filtra per
            # job_id e stato), ma se in futuro qualcosa dovesse leggerlo, i due
            # ambienti restituirebbero risultati diversi. Scriviamo entrambe le
            # chiavi con lo stesso valore: costa un attributo e rende i record
            # AWS e locali interpretabili nello stesso modo.
            "worker_name": worker_id,
            "worker_id": worker_id,
        }
        self._db.put_item(WORKER_TASKS_TABLE, task_id, payload)

    def update_worker_task_status(self, job_id: str, worker_id: str, status: str) -> None:
        task_id = f"{job_id}#{worker_id}"
        current_item = self._unwrap(self._db.get_item(WORKER_TASKS_TABLE, task_id))
        current_item.update({"status": status, "timestamp": int(time.time())})
        self._db.put_item(WORKER_TASKS_TABLE, task_id, current_item)

    def are_all_workers_done(self, job_id: str, expected_count: int) -> bool:
        """
        Verifica se tutti i worker attesi hanno completato la propria parte del job.
        """
        response = self._db.query_by_index(
            table_name=WORKER_TASKS_TABLE,
            index_name=WORKER_TASKS_JOB_INDEX,
            key_name="job_id",
            key_value=job_id,
        )

        job_tasks = response.get("Items", [])
        completed_tasks = [t for t in job_tasks if t.get("status") == "COMPLETED"]
        return len(completed_tasks) == expected_count

    def get_active_jobs(self) -> list:
        response = self._db.scan_table(JOBS_TABLE)
        all_jobs = response.get("Items", [])
        return [j.get("job_id") for j in all_jobs if j.get("status") == "PROCESSING" and j.get("job_id")]

    def get_job_status(self, job_id: str) -> Optional[str]:
        item = self._unwrap(self._db.get_item(JOBS_TABLE, job_id))
        return item.get("status") if item else None

    def get_job_details(self, job_id: str) -> Optional[dict]:
        item = self._unwrap(self._db.get_item(JOBS_TABLE, job_id))
        return item if item else None

    def acquire_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        return self._db.try_acquire_lock(LOCKS_TABLE, lock_key, owner, ttl)

    def refresh_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        return self._db.refresh_lock(LOCKS_TABLE, lock_key, owner, ttl)

    def release_global_lock(self, lock_key: str, owner: str) -> bool:
        return self._db.release_lock(LOCKS_TABLE, lock_key, owner)

    def try_claim_job(self, job_id: str, orchestrator_id: str, lease_seconds: int = 300) -> bool:
        claimed = self._db.try_acquire_lock(JOB_LOCKS_TABLE, job_id, orchestrator_id, ttl=lease_seconds)
        if not claimed:
            claimed = self._db.refresh_lock(JOB_LOCKS_TABLE, job_id, orchestrator_id, ttl=lease_seconds)

        if not claimed:
            print(f"[AWS StateManager] [CLAIM FAILED] Job {job_id[:8]}... già posseduto da un altro Orchestrator.")
        return claimed

    def release_job_lease(self, job_id: str, orchestrator_id: str) -> bool:
        released = self._db.release_lock(JOB_LOCKS_TABLE, job_id, orchestrator_id)
        if released:
            print(f"[AWS StateManager] Lease rilasciata per Job ID: {job_id[:8]}... da {orchestrator_id}")
        return released