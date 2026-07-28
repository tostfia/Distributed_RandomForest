import os
import requests
from typing import Optional
from src.shared.mock_aws.interfaces import SQSQueueInterface


class LambdaGatewaySQSQueue(SQSQueueInterface):
    """
    Implementazione dell'interfaccia SQS che, invece di parlare direttamente
    con Amazon SQS via boto3, passa per API Gateway -> Lambda -> SQS.

    NOTA: questa classe è pensata per il lato CLIENT (producer) del sistema,
    che deve solo INVIARE job. Il consumo (receive/delete) resta compito degli
    orchestratori/worker lato server, che continuano a usare AwsSQSQueue con
    accesso diretto a boto3 dentro l'infrastruttura AWS (VPC/ECS/EC2).

    Richiede la variabile d'ambiente API_GATEWAY_URL (nel file .env):
        API_GATEWAY_URL=https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com
    """

    def __init__(self, base_url: str = None):
        url = base_url or os.environ.get("API_GATEWAY_URL")
        if not url:
            raise ValueError(
                "API_GATEWAY_URL non impostata. Aggiungila al file .env "
                "puntando all'Invoke URL della tua HTTP API."
            )
        self.base_url = url.rstrip("/")

    def send_message(self, queue_name: str, message_dict: dict) -> None:
        path = "federated" if "federated" in queue_name else "centralized"
        url = f"{self.base_url}/jobs/{path}"

        response = requests.post(url, json=message_dict, timeout=10)
        response.raise_for_status()

        body = response.json()
        return {"MessageId": body.get("message_id"), "job_id": body.get("job_id")}

    def receive_message(self, queue_name: str, visibility_timeout: int = 300) -> Optional[dict]:
        raise NotImplementedError(
            "LambdaGatewaySQSQueue è un client 'solo invio' (producer) che passa da "
            "API Gateway. Il consumo dei messaggi va fatto con AwsSQSQueue lato "
            "orchestratore/worker, che ha accesso diretto a SQS."
        )

    def delete_message(self, receipt_handle: str) -> bool:
        raise NotImplementedError(
            "LambdaGatewaySQSQueue è un client 'solo invio' (producer) che passa da "
            "API Gateway. Il consumo dei messaggi va fatto con AwsSQSQueue lato "
            "orchestratore/worker, che ha accesso diretto a SQS."
        )