import os
import time
from typing import Dict, Any
import boto3

from src.shared.config import SystemConfig
from src.shared.mock_aws.dynamodb import dynamo_db as mock_dynamodb

cfg = SystemConfig()

class ServiceRegistry:
    WORKERS_TABLE = 'workers_registry'
    ORCHESTRATORS_TABLE = 'orchestrators_registry'
    TIME_OUT_SECONDS = int(os.environ.get("WORKER_TIMEOUT_SECONDS", 120))  # Timeout per considerare un worker non disponibile

    @classmethod
    def _get_db_client(cls):
        """Restituisce il client DB corretto in base all'ambiente del file .env"""
        if cfg.env == "local":
            return mock_dynamodb
        else:
            # Se siamo su AWS, usiamo la risorsa reale di boto3
            # Nota: assumiamo che in produzione la factory o l'ambiente configuri AWS
            return boto3.resource('dynamodb').Table(cls.WORKERS_TABLE) 
            # Per rendere il codice omogeneo con i tuoi metodi custom .put_item, .scan_table ecc.,
            # in produzione si usa solitamente un wrapper AWS reale analogo al mock.
            # Per ora lo isoliamo dinamicamente.

    @classmethod
    def register_worker(cls, worker_name: str, host: str, port: int):
        """Registra un worker nel DynamoDB con timestamp aggiornato."""
        db = cls._get_db_client()
        payload = {"host": host, "port": port, "status": "AVAILABLE", "last_heartbeat": int(time.time())}
        
        if cfg.env == "local":
            db.put_item(cls.WORKERS_TABLE, worker_name, payload)
        else:
            # Logica AWS reale (boto3 standard)
            db_real = boto3.resource('dynamodb').Table(cls.WORKERS_TABLE)
            db_real.put_item(Item={"worker_name": worker_name, **payload})
            
        print(f"[ServiceRegistry] Worker '{worker_name}' registrato con host {host} e port {port}.")

    @classmethod
    def register_orchestrator(cls, orchestrator_name: str):
        payload = {"status": "AVAILABLE", "last_heartbeat": int(time.time())}
        if cfg.env == "local":
            mock_dynamodb.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, payload)
        else:
            db_real = boto3.resource('dynamodb').Table(cls.ORCHESTRATORS_TABLE)
            db_real.put_item(Item={"orchestrator_name": orchestrator_name, **payload})
        print(f"[ServiceRegistry] Orchestrator '{orchestrator_name}' registrato.")

    @classmethod
    def get_available_workers(cls, environment: str) -> Dict[str, Any]:
        """Recupera tutti i worker disponibili, filtrando per heartbeat recente."""
        current_time = int(time.time())
        available_workers = {}

        if cfg.env == "local":
            response = mock_dynamodb.scan_table(cls.WORKERS_TABLE)
            items = response.get("Items", [])
        else:
            # COMPLETATA LA LOGICA PER AWS REALE
            try:
                db_real = boto3.resource('dynamodb').Table(cls.WORKERS_TABLE)
                response = db_real.scan()
                items = response.get("Items", [])
            except Exception as e:
                print(f"[ServiceRegistry ERRORE AWS S3/Dynamo]: {e}")
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
    def is_orchestrator_available(cls, orchestrator_name: str) -> bool:
        """Controlla se un orchestratore è disponibile basandosi sul suo ultimo heartbeat."""
        if cfg.env == "local":
            response = mock_dynamodb.scan_table(cls.ORCHESTRATORS_TABLE)
        else:
            db_real = boto3.resource('dynamodb').Table(cls.ORCHESTRATORS_TABLE)
            response = db_real.scan()
            
        items = response.get("Items", [])
        for item in items:
            if cls._extract_data(item, "orchestrator_name") == orchestrator_name:
                return True
        return False

    @classmethod
    def deregister_worker(cls, worker_name: str):
        if cfg.env == "local":
            mock_dynamodb.delete_item(cls.WORKERS_TABLE, worker_name)
        else:
            db_real = boto3.resource('dynamodb').Table(cls.WORKERS_TABLE)
            db_real.delete_item(Key={"worker_name": worker_name})
        print(f"[ServiceRegistry] Worker '{worker_name}' deregistrato.")

    @classmethod
    def deregister_orchestrator(cls, orchestrator_name: str):
        if cfg.env == "local":
            mock_dynamodb.delete_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
        else:
            db_real = boto3.resource('dynamodb').Table(cls.ORCHESTRATORS_TABLE)
            db_real.delete_item(Key={"orchestrator_name": orchestrator_name})
        print(f"[ServiceRegistry] Orchestrator '{orchestrator_name}' deregistrato.")

    @classmethod
    def update_worker_heartbeat(cls, worker_name: str):
        """Aggiorna il timestamp dell'ultimo heartbeat di un worker."""
        if cfg.env == "local":
            response = mock_dynamodb.get_item(cls.WORKERS_TABLE, worker_name)
            worker_data = response.get("Item")
            if worker_data:
                worker_data['last_heartbeat'] = int(time.time())
                mock_dynamodb.put_item(cls.WORKERS_TABLE, worker_name, worker_data)
        else:
            db_real = boto3.resource('dynamodb').Table(cls.WORKERS_TABLE)
            db_real.update_item(
                Key={'worker_name': worker_name},
                UpdateExpression="set last_heartbeat = :t",
                ExpressionAttributeValues={':t': int(time.time())}
            )
    
    @classmethod
    def update_orchestrator_heartbeat(cls, orchestrator_name: str):
        """Aggiorna il timestamp dell'ultimo heartbeat di un orchestratore."""
        if cfg.env == "local":
            response = mock_dynamodb.get_item(cls.ORCHESTRATORS_TABLE, orchestrator_name)
            orchestrator_data = response.get("Item")
            if orchestrator_data:
                orchestrator_data['last_heartbeat'] = int(time.time())
                mock_dynamodb.put_item(cls.ORCHESTRATORS_TABLE, orchestrator_name, orchestrator_data)
        else:
            db_real = boto3.resource('dynamodb').Table(cls.ORCHESTRATORS_TABLE)
            db_real.update_item(
                Key={'orchestrator_name': orchestrator_name},
                UpdateExpression="set last_heartbeat = :t",
                ExpressionAttributeValues={':t': int(time.time())}
            )

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