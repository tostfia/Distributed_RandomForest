"""
Implementazione reale della coda tramite Amazon SQS (boto3).

Espone la STESSA interfaccia pubblica di MockSQSQueue (sqs.py):
    - send_message(queue_name, message_dict) -> None
    - receive_message(queue_name, visibility_timeout=300) -> {"Body": dict, "ReceiptHandle": str} | None
    - delete_message(receipt_handle) -> bool
    - change_message_visibility(queue_name, receipt_handle, visibility_timeout) -> bool

Nota importante su delete_message
----------------------------------
L'API reale `sqs:DeleteMessage` richiede l'URL della coda, non solo il
ReceiptHandle. L'interfaccia esistente però chiama
`self.sqs_queue.delete_message(receipt_handle)` da BaseOrchestrator senza
passare il queue_name. Per non dover toccare l'interfaccia e tutto il
codice che la usa, manteniamo una mappa interna
    receipt_handle -> queue_url
popolata ad ogni receive_message() e consumata da delete_message() /
change_message_visibility(). Questo è sicuro perché ogni orchestratore fa
polling su UNA sola coda alla volta (self.queue_name), quindi non c'è
ambiguità su quale coda appartenga un dato ReceiptHandle.
"""

import json
import os
import uuid
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.shared.mock_aws.interfaces import SQSQueueInterface


class AwsSQSQueue(SQSQueueInterface):

    def __init__(self, region_name: Optional[str] = None):
        region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("sqs", region_name=region_name)
        self._queue_url_cache: dict[str, str] = {}
        self._receipt_to_queue: dict[str, str] = {}

    def _resolve_queue_url(self, queue_name: str) -> str:
        if queue_name not in self._queue_url_cache:
            try:
                response = self._client.get_queue_url(QueueName=queue_name)
            except ClientError as e:
                raise RuntimeError(
                    f"[AWS SQS] Coda '{queue_name}' non trovata su AWS. "
                    f"Va creata (vedi setup_aws_resources.py) prima di avviare il sistema con SYS_ENV=aws."
                ) from e
            self._queue_url_cache[queue_name] = response["QueueUrl"]
        return self._queue_url_cache[queue_name]

    def send_message(self, queue_name: str, message_dict: dict,**kwargs) -> None:
        if "job_id" not in message_dict:
            raise ValueError("[AWS SQS]: Il messaggio deve contenere un 'job_id' univoco.")

        queue_url = self._resolve_queue_url(queue_name)
        
        # Assegnazione dinamica del Group ID in base alla coda di destinazione
        if "federated" in queue_name:
            group_id = "ML-Federated-Group"
        else:
            group_id = "ML-Centralized-Group"

        send_params = {
            "QueueUrl": queue_url,
            "MessageBody": json.dumps(message_dict),
        }

        if queue_name.endswith(".fifo"):
            send_params["MessageGroupId"] = group_id
            send_params["MessageDeduplicationId"] = str(uuid.uuid4())
            log_tipo_coda = "[FIFO]"
        else:
            log_tipo_coda = "[STANDARD]"
        
        self._client.send_message(**send_params)
        print(f"[AWS SQS] {log_tipo_coda} Messaggio inviato in '{queue_name}' - Job ID: {message_dict['job_id'][:8]}...")
        
       
    def receive_message(self, queue_name: str, visibility_timeout: int = 300) -> Optional[dict]:
        queue_url = self._resolve_queue_url(queue_name)

        response = self._client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            VisibilityTimeout=visibility_timeout,
            WaitTimeSeconds=10,  # long polling: meno chiamate a vuoto, meno costi
        )

        messages = response.get("Messages")
        if not messages:
            return None

        raw = messages[0]
        receipt_handle = raw["ReceiptHandle"]
        body = json.loads(raw["Body"])

        # Ricordiamo a quale coda appartiene questo handle, per poterlo
        # cancellare/estendere senza dover ripassare il queue_name.
        self._receipt_to_queue[receipt_handle] = queue_url

        return {
            "Body": body,
            "ReceiptHandle": receipt_handle,
        }

    def delete_message(self, receipt_handle: str) -> bool:
        queue_url = self._receipt_to_queue.pop(receipt_handle, None)
        if queue_url is None:
            print(f"[AWS SQS] [ERRORE CANCELLAZIONE] ReceiptHandle sconosciuto o già elaborato: {receipt_handle}")
            return False
        try:
            self._client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            print(f"[AWS SQS] [OK] Messaggio eliminato tramite ReceiptHandle: {receipt_handle}")
            return True
        except ClientError as e:
            print(f"[AWS SQS] [ERRORE CANCELLAZIONE] {e}")
            return False

    def change_message_visibility(self, queue_name: str, receipt_handle: str, visibility_timeout: int) -> bool:
        queue_url = self._receipt_to_queue.get(receipt_handle) or self._resolve_queue_url(queue_name)
        try:
            self._client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout,
            )
            print(f"[AWS SQS] [HEARTBEAT OK] Visibilità estesa per {receipt_handle} di altri {visibility_timeout}s.")
            return True
        except ClientError as e:
            print(f"[AWS SQS] [HEARTBEAT WARN] Impossibile estendere la visibilità: {e}")
            return False