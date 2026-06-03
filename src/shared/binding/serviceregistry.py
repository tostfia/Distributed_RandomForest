import time
from typing import Dict, Any

from src.shared.mock_aws.dynamodb import dynamo_db as dynamodb

class ServiceRegistry:
    WORKERS_TABLE = 'workers_registry'
    ORCHESTRATORS_TABLE = 'orchestrators_registry'
    TIME_OUT_SECONDS = 60

    @classmethod
    def register_worker(cls, worker_name: str, host: str, port: int):
        """Registra un worker nel DynamoDB con timestamp aggiornato."""
        payload = {"host": host, "port": port, "status": "AVAILABLE", "last_heartbeat": int(time.time())}
        dynamodb.put_item(cls.WORKERS_TABLE, worker_name, payload)
        print(f"[ServiceRegistry] Worker '{worker_name}' registrato con host {host} e port {port}.")

    @classmethod
    def register_orchestrator(cls, orchestrator_name: str):
        payload = {"status": "AVAILABLE", "last_heartbeat": int(time.time())}
        dynamodb.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, payload)
        print(f"[ServiceRegistry] Orchestrator '{orchestrator_name}' registrato.")

    @classmethod
    def get_available_workers(cls, environment: str) -> Dict[str, Any]:
        """Recupera tutti i worker disponibili, filtrando quelli che non hanno inviato heartbeat recentemente."""
        current_time = int(time.time())
        available_workers = {}

        if environment == "local":
            response = dynamodb.scan_table(cls.WORKERS_TABLE)
        else:
            ##Comportamento da implementare per il distribuito con AWS
            pass
        items = response.get("Items", [])
        for item in items:
            worker_name = item.get("worker_name")
            last_heartbeat = item.get("last_heartbeat", 0)
            if current_time - last_heartbeat <= cls.TIME_OUT_SECONDS:
                available_workers[worker_name] = {
                    "host": item.get("host"),
                    "port": item.get("port"),
                    "status": item.get("status"),
                    "last_heartbeat": last_heartbeat
                }
        return available_workers

    @classmethod
    def is_orchestrator_available(cls, orchestrator_name: str) -> bool:
        """Controlla se un orchestratore è disponibile basandosi sul suo ultimo heartbeat."""
        response = dynamodb.scan_table(cls.ORCHESTRATORS_TABLE)
        items = response.get("Items", [])
        for item in items:
            if item.get("orchestrator_name") == orchestrator_name:
                return True
        return False

    @classmethod
    def deregister_worker(cls, worker_name: str):
        dynamodb.delete_item(cls.WORKERS_TABLE, worker_name)
        print(f"[ServiceRegistry] Worker '{worker_name}' deregistrato.")

    @classmethod
    def deregister_orchestrator(cls, orchestrator_name: str):
        dynamodb.delete_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
        print(f"[ServiceRegistry] Orchestrator '{orchestrator_name}' deregistrato.")

    @classmethod
    def update_worker_heartbeat(cls, worker_name: str):
        """Aggiorna il timestamp dell'ultimo heartbeat di un worker."""
        worker = dynamodb.get_item(cls.WORKERS_TABLE, worker_name)
        if worker:
            worker['last_heartbeat'] = int(time.time())
            dynamodb.put_item(cls.WORKERS_TABLE, worker_name, worker)
            print(f"[ServiceRegistry] Heartbeat aggiornato per worker '{worker_name}'.")
    
    @classmethod
    def update_orchestrator_heartbeat(cls, orchestrator_name: str):
        """Aggiorna il timestamp dell'ultimo heartbeat di un orchestratore."""
        orchestrator = dynamodb.get_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
        if orchestrator:
            orchestrator['last_heartbeat'] = int(time.time())
            dynamodb.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, orchestrator)
            print(f"[ServiceRegistry] Heartbeat aggiornato per orchestratore '{orchestrator_name}'.")
