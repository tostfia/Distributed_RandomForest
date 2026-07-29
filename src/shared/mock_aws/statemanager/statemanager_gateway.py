import os
import requests
from typing import Optional
from src.shared.mock_aws.interfaces import StateManagerInterface

class ApiGatewayStateManager(StateManagerInterface):
    """
    Implementazione dello StateManager specifica per il CLIENT in ambiente AWS.
    Segue i principi Zero-Trust: non usa boto3 né credenziali dirette su DynamoDB,
    ma passa per API Gateway -> Lambda.
    """
    def __init__(self, base_url: str = None):
        url = base_url or os.environ.get("API_GATEWAY_URL")
        if not url:
            raise ValueError(
                "API_GATEWAY_URL non impostata. Aggiungila al file .env "
                "puntando all'Invoke URL della tua HTTP API (es. https://c9ao92zrdh.execute-api.us-east-1.amazonaws.com)"
            )
        self.base_url = url.rstrip("/")

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Invia una richiesta GET ad API Gateway per conoscere lo stato del job."""
        url = f"{self.base_url}/jobs/{job_id}/status"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Il corpo JSON restituito dalla Lambda (es. {"status": "COMPLETED", ...})
                return data.get("status")
            elif response.status_code == 404:
                return None
            else:
                print(f"[ApiGatewayStateManager] Errore API Gateway ({response.status_code}): {response.text}")
                return None
        except requests.RequestException as e:
            print(f"[ApiGatewayStateManager] Chiamata HTTP fallita: {e}")
            return None

    # I metodi sottostanti non sono usati dal Client ma devono essere presenti
    # per rispettare l'interfaccia StateManagerInterface
    def initiate_request(self, job_id: str, dataset_path: str, seed: int) -> None:
        pass  # In AWS la registrazione della richiesta è gestita dalla POST inviata a LambdaGatewaySQSQueue

    def obtain_request(self, job_id: str) -> Optional[dict]:
        raise NotImplementedError("Il client interroga lo stato solo tramite get_job_status.")

    def update_request_status(self, job_id: str, status: str, orchestrator_id: str, **kwargs) -> None:
        raise NotImplementedError("Operazione riservata agli orchestrator server-side.")

    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        raise NotImplementedError("Operazione riservata agli orchestrator server-side.")

    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        raise NotImplementedError("Operazione riservata agli orchestrator/worker.")

    def update_worker_task_status(self, job_id: str, worker_id: str, status: str) -> None:
        raise NotImplementedError("Operazione riservata agli orchestrator/worker.")

    def are_all_workers_done(self, job_id: str, expected_count: int) -> bool:
        raise NotImplementedError("Operazione riservata agli orchestrator/worker.")

    def get_active_jobs(self) -> list:
        raise NotImplementedError("Operazione riservata agli orchestrator.")

    def acquire_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        raise NotImplementedError("Operazione riservata agli orchestrator.")

    def refresh_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        raise NotImplementedError("Operazione riservata agli orchestrator.")

    def release_global_lock(self, lock_key: str, owner: str) -> bool:
        raise NotImplementedError("Operazione riservata agli orchestrator.")

    def try_claim_job(self, job_id: str, orchestrator_id: str, lease_seconds: int = 300) -> bool:
        raise NotImplementedError("Operazione riservata agli orchestrator.")

    def release_job_lease(self, job_id: str, orchestrator_id: str) -> bool:
        raise NotImplementedError("Operazione riservata agli orchestrator.")