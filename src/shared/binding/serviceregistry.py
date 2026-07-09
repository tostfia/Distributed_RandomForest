import os
import time
from typing import Dict, Any
import boto3

from src.shared.mock_aws.dynamodb.dynamodb_factory import DynamoDBFactory
from src.shared.config import SystemConfig
from src.shared.mock_aws.dynamodb.dynamodb import dynamo_db as mock_dynamodb

cfg = SystemConfig()

class ServiceRegistry:
    WORKERS_TABLE = 'workers_registry'
    ORCHESTRATORS_TABLE = 'orchestrators_registry'
    TIME_OUT_SECONDS = int(os.environ.get("WORKER_TIMEOUT_SECONDS", 120))  # Timeout per considerare un worker non disponibile

    @classmethod
    def _get_db_client(cls):
        """Restituisce il client DB corretto in base all'ambiente del file .env"""
        return DynamoDBFactory.get_db(cfg.env)

    @classmethod
    def register_worker(cls, worker_name: str, host: str, port: int):
        """Registra un worker nel DynamoDB con timestamp aggiornato."""
        db = cls._get_db_client()
        payload = {"host": host, "port": port, "status": "AVAILABLE", "last_heartbeat": int(time.time())}
        
        db.put_item(cls.WORKERS_TABLE, worker_name, payload)
            
        print(f"[ServiceRegistry] Worker '{worker_name}' registrato con host {host} e port {port}.")

    @classmethod
    def register_orchestrator(cls, orchestrator_name: str):
        payload = {"status": "AVAILABLE", "last_heartbeat": int(time.time())}
        db = cls._get_db_client()
        db.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, payload)
        print(f"[ServiceRegistry] Orchestrator '{orchestrator_name}' registrato.")

    @classmethod
    def get_available_workers(cls, environment: str) -> Dict[str, Any]:
        """Recupera tutti i worker disponibili, filtrando per heartbeat recente."""
        current_time = int(time.time())
        available_workers = {}

        db = cls._get_db_client()
        try:
            response = db.scan_table(cls.WORKERS_TABLE)
            items = response.get("Items", [])
        except Exception as e:
            print(f"[ServiceRegistry ERRORE DynamoDB]: {e}")
            return {}

        for item in items:
            worker_name = cls._extract_data(item, "worker_name")
            if not worker_name:
                continue 
            last_heartbeat = int(cls._extract_data(item, "last_heartbeat") or 0)
            
            if current_time - last_heartbeat <= cls.TIME_OUT_SECONDS:
                available_workers[worker_name] = {
                    "host": cls._extract_data(item, "host"),
                    "port": cls._extract_data(item, "port"),
                    "status": cls._extract_data(item, "status"),
                    "last_heartbeat": last_heartbeat
                }
        return available_workers
    
    @classmethod
    def get_expired_workers(cls) -> Dict[str, Any]:
        """Recupera tutti i worker scaduti (non disponibili)."""
        current_time = int(time.time())
        expired_workers = {}

        db = cls._get_db_client()
        try:
            response = db.scan_table(cls.WORKERS_TABLE)
            items = response.get("Items", [])
        except Exception as e:
            print(f"[ServiceRegistry ERRORE DynamoDB]: {e}")
            return {}

        for item in items:
            worker_name = cls._extract_data(item, "worker_name")
            if not worker_name:
                continue 
            last_heartbeat = int(cls._extract_data(item, "last_heartbeat") or 0)
            
            if current_time - last_heartbeat > cls.TIME_OUT_SECONDS:
                expired_workers[worker_name] = {
                    "host": cls._extract_data(item, "host"),
                    "port": cls._extract_data(item, "port"),
                    "status": cls._extract_data(item, "status"),
                    "last_heartbeat": last_heartbeat,
                    "second_since_heartbeat": current_time - last_heartbeat
                }
        return expired_workers

    @classmethod
    def is_orchestrator_available(cls, orchestrator_name: str) -> bool:
        """Controlla se un orchestratore è disponibile basandosi sul suo ultimo heartbeat."""
        db = cls._get_db_client()
        response = db.scan_table(cls.ORCHESTRATORS_TABLE)
        items = response.get("Items", [])
        for item in items:
            if cls._extract_data(item, "orchestrator_name") == orchestrator_name:
                return True
        return False

    @classmethod
    def deregister_worker(cls, worker_name: str):
        db = cls._get_db_client()
        db.delete_item(cls.WORKERS_TABLE, worker_name)
        print(f"[ServiceRegistry] Worker '{worker_name}' deregistrato.")

    @classmethod
    def deregister_orchestrator(cls, orchestrator_name: str):
        db = cls._get_db_client()
        db.delete_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
            
        print(f"[ServiceRegistry] Orchestrator '{orchestrator_name}' deregistrato.")

    @classmethod
    def update_worker_heartbeat(cls, worker_name: str):
        """Aggiorna il timestamp dell'ultimo heartbeat di un worker."""
        db = cls._get_db_client()
        response = db.get_item(cls.WORKERS_TABLE, worker_name)
        worker_data = response.get("Item")
        if worker_data:
            worker_data['last_heartbeat'] = int(time.time())
            db.put_item(cls.WORKERS_TABLE, worker_name, worker_data)
    
    @classmethod
    def update_orchestrator_heartbeat(cls, orchestrator_name: str):
        """Aggiorna il timestamp dell'ultimo heartbeat di un orchestratore."""
        db = cls._get_db_client()
        response = db.get_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
        orchestrator_data = response.get("Item")
        if orchestrator_data:
            orchestrator_data['last_heartbeat'] = int(time.time())
            db.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, orchestrator_data)

    @classmethod
    def _extract_data(cls, data: dict, key: str):
        """Versione sicura e piatta per estrarre chiavi."""
        if not isinstance(data, dict):
            return None
        if key in data:
            return data[key]
        if "Item" in data and isinstance(data["Item"], dict):
            return data["Item"].get(key)
        return None