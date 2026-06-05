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
            return {}  # Aggiungere logica per AWS 
        items = response.get("Items", [])
        for item in items:
            worker_name = cls._extract_data(item, "worker_name")
            if not worker_name:
                continue 
            last_heartbeat = cls._extract_data(item, "last_heartbeat") or 0
            
            if current_time - last_heartbeat <= cls.TIME_OUT_SECONDS:
                available_workers[worker_name] = {
                    "host": cls._extract_data(item, "host"),
                    "port": cls._extract_data(item, "port"),
                    "status": cls._extract_data(item, "status"),
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
        response = dynamodb.get_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
        
        # Estrai i dati reali dal wrapper "Item"
        orchestrator_data = response.get("Item")
        
        if orchestrator_data:
            # Aggiorna solo il dizionario dei dati, non il wrapper
            orchestrator_data['last_heartbeat'] = int(time.time())
            
            # Passa solo il dizionario dei dati a put_item
            dynamodb.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, orchestrator_data)
            print(f"[ServiceRegistry] Heartbeat aggiornato per orchestratore '{orchestrator_name}'.")

    @classmethod
    def _extract_data(cls, data: dict, key: str):
        """Funzione ricorsiva per trovare una chiave in un dizionario annidato."""
        if key in data:
            return data[key]
        for k, v in data.items():
            if isinstance(v, dict):
                res = cls._extract_data(v, key)
                if res: return res
        return None
