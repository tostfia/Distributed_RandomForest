"""
Query di lettura su WorkerTasks tramite i due GSI (job_id-index,
worker_name-index) invece di uno scan completo. Stessa interfaccia in
locale (filtro in memoria) e su AWS (query reale sul GSI).
"""

from typing import Dict, Any, List

from src.shared.config import SystemConfig
from src.shared.mock_aws.dynamodb.dynamodb_factory import DynamoDBFactory

cfg = SystemConfig()


class TaskRegistry:
    TASKS_TABLE = "WorkerTasks"

    @classmethod
    def _get_db_client(cls):
        return DynamoDBFactory.get_db(cfg.env)

    @classmethod
    def get_tasks_by_job(cls, job_id: str) -> List[Dict[str, Any]]:
        """Tutti i task di un job, via GSI job_id-index (o filtro equivalente in locale)."""
        db = cls._get_db_client()
        response = db.query_by_index(cls.TASKS_TABLE, "job_id-index", "job_id", job_id)
        return response.get("Items", [])

    @classmethod
    def get_tasks_by_worker(cls, worker_name: str) -> List[Dict[str, Any]]:
        """Tutti i task assegnati a un worker, via GSI worker_name-index."""
        db = cls._get_db_client()
        response = db.query_by_index(cls.TASKS_TABLE, "worker_name-index", "worker_name", worker_name)
        return response.get("Items", [])