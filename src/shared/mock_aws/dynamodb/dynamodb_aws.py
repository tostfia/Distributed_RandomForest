"""
Implementazione reale di accesso a DynamoDB tramite boto3.
Espone la STESSA interfaccia pubblica di MockDynamoDB (dynamodb.py) in modo
da poter essere scambiata "a caldo" con quella mock semplicemente cambiando
l'oggetto restituito da una factory (vedi dynamodb_factory.py).

Interfaccia mantenuta identica:
    - put_item(table_name, key, value)
    - put_item_if_not_exists(table_name, key, value) -> bool
    - get_item(table_name, key) -> {"Item": {...}} oppure {}
    - delete_item(table_name, key) -> bool
    - scan_table(table_name) -> {"Items": [...]}
"""

import os
import decimal
import threading
import time
from typing import Optional


import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key


class AwsDynamoDB:

    _PK_MAPPING = {
        'workers_registry': 'worker_name',
        'orchestrators_registry': 'orchestrator_name',
        'ModelStatus': 'job_id',
        'WorkerTasks': 'task_id',
        'OrchestratorLocks': 'lock_key',
        'JobLocks': 'lock_key',
        'WorkerIndexLocks': 'lock_key'
    }

    def __init__(self, region_name: Optional[str] = None):
        
        
        self._region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._thread_local = threading.local()
      

    # ------------------------------------------------------------------
    # Utility interne
    # ------------------------------------------------------------------

    def _get_primary_key_name(self, table_name: str) -> str:
        if table_name not in self._PK_MAPPING:
            raise ValueError(f"Tabella '{table_name}' non riconosciuta.")
        return self._PK_MAPPING[table_name]

    def _get_resource(self):
        if not hasattr(self._thread_local, "resource"):
            self._thread_local.resource = boto3.resource("dynamodb", region_name=self._region_name)
        return self._thread_local.resource

    def _table(self, table_name: str):
        if table_name not in self._PK_MAPPING:
            raise ValueError(f"Tabella '{table_name}' non riconosciuta.")
        if not hasattr(self._thread_local, "tables"):
            self._thread_local.tables = {}
        if table_name not in self._thread_local.tables:
            self._thread_local.tables[table_name] = self._get_resource().Table(table_name)
        return self._thread_local.tables[table_name]

    @staticmethod
    def _to_dynamo(value):
        """DynamoDB non accetta float nativi: vanno convertiti in Decimal.
        Applica la conversione ricorsivamente su dict/list."""
        if isinstance(value, float):
            # str() evita i classici problemi di precisione binaria di Decimal(float)
            return decimal.Decimal(str(value))
        if isinstance(value, dict):
            return {k: AwsDynamoDB._to_dynamo(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AwsDynamoDB._to_dynamo(v) for v in value]
        return value

    @staticmethod
    def _from_dynamo(value):
        """Riconverte i Decimal restituiti da DynamoDB in int/float nativi Python."""
        if isinstance(value, decimal.Decimal):
            return int(value) if value % 1 == 0 else float(value)
        if isinstance(value, dict):
            return {k: AwsDynamoDB._from_dynamo(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AwsDynamoDB._from_dynamo(v) for v in value]
        return value

    # ------------------------------------------------------------------
    # API pubblica (stessa firma del mock)
    # ------------------------------------------------------------------

    def put_item(self, table_name: str, key: str, value: dict):
        pk_name = self._get_primary_key_name(table_name)
        item = self._to_dynamo(dict(value))
        item[pk_name] = str(key)
        self._table(table_name).put_item(Item=item)

    def put_item_if_not_exists(self, table_name: str, key: str, value: dict) -> bool:
        pk_name = self._get_primary_key_name(table_name)
        item = self._to_dynamo(dict(value))
        item[pk_name] = str(key)
        try:
            self._table(table_name).put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(#pk)",
                ExpressionAttributeNames={"#pk": pk_name},
            )
            print(f"[AWS DynamoDB] Tabella '{table_name}' -> Conditional write OK: {str(key)[:8]}...")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                print(f"[AWS DynamoDB] Tabella '{table_name}' -> Conditional write FALLITA (già esiste): {str(key)[:8]}...")
                return False
            raise

    def get_item(self, table_name: str, key: str) -> Optional[dict]:
        pk_name = self._get_primary_key_name(table_name)
        response = self._table(table_name).get_item(Key={pk_name: str(key)})
        item = response.get("Item")
        if item:
            return {"Item": self._from_dynamo(item)}
        return {}

    def delete_item(self, table_name: str, key: str) -> bool:
        pk_name = self._get_primary_key_name(table_name)
        try:
            self._table(table_name).delete_item(
                Key={pk_name: str(key)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": pk_name},
            )
            print(f"[AWS DynamoDB] Tabella '{table_name}' -> Eliminato ID: {str(key)[:8]}...")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def scan_table(self, table_name: str) -> dict:
        table = self._table(table_name)
        items = []
        response = table.scan()
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))
        return {"Items": [self._from_dynamo(i) for i in items]}
    
    def query_by_index(self, table_name: str, index_name: str, key_name: str, key_value: str) -> dict:
        """
        Esegue una query su un Global Secondary Index (GSI).
        """
        table = self._table(table_name)
        items = []
        
        # Eseguiamo la prima query sull'indice
        response = table.query(
            IndexName=index_name,
            KeyConditionExpression=Key(key_name).eq(str(key_value))
        )
        items.extend(response.get("Items", []))
        
        # Gestione dell'impaginazione (limite di 1MB di DynamoDB)
        while "LastEvaluatedKey" in response:
            response = table.query(
                IndexName=index_name,
                KeyConditionExpression=Key(key_name).eq(str(key_value)),
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            items.extend(response.get("Items", []))
            
        return {"Items": [self._from_dynamo(i) for i in items]}

    def try_acquire_lock(self, table_name: str, lock_key: str, owner: str, ttl: int = 30) -> bool:
        now = int(time.time())
        expires_at = now + int(ttl)
        table = self._table(table_name)

        try:
            table.put_item(
                Item={
                    "lock_key": lock_key,
                    "leader": owner,
                    "expires_at": expires_at,
                    "timestamp": now,
                },
                ConditionExpression="attribute_not_exists(lock_key) OR expires_at < :now",
                ExpressionAttributeValues={":now": now},
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    
    def refresh_lock(self, table_name: str, lock_key: str, owner: str, ttl: int = 30) -> bool:
        now = int(time.time())
        expires_at = now + int(ttl)
        table = self._table(table_name)

        try:
            table.update_item(
                Key={"lock_key": lock_key},
                UpdateExpression="SET expires_at = :exp, #ts = :ts",
                ConditionExpression="leader = :owner",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={
                    ":owner": owner,
                    ":exp": expires_at,
                    ":ts": now,
                },
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
    
    def release_lock(self, table_name: str, lock_key: str, owner: str) -> bool:
        table = self._table(table_name)
        try:
            table.delete_item(
                Key={"lock_key": lock_key},
                ConditionExpression="leader = :owner",
                ExpressionAttributeValues={":owner": owner},
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

aws_dynamo_db = AwsDynamoDB()