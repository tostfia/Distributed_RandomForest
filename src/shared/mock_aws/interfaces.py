from abc import ABC, abstractmethod
from typing import Optional

class SQSQueueInterface(ABC):
    @abstractmethod
    def send_message(self, queue_name: str, message_dict: dict) -> None:
        """Invia un messaggio a una coda specifica."""
        pass

    @abstractmethod
    def receive_message(self, queue_name: str, visibility_timeout: int = 30) -> Optional[dict]:
        """Fa polling e legge un messaggio da una coda specifica."""
        pass

    @abstractmethod
    def delete_message(self, receipt_handle: str) -> bool:
        """Elimina un messaggio in-flight tramite il suo Receipt Handle."""
        pass


class StateManagerInterface(ABC):
    @abstractmethod
    def initiate_request(self, job_id: str, dataset_path: str) -> None:
        """Registra la richiesta iniziale nel sistema di tracciamento dello stato."""
        pass

    @abstractmethod
    def obtain_request(self, job_id: str) -> Optional[dict]:
        """Recupera lo stato corrente di un determinato Job."""
        pass

    @abstractmethod
    def update_request_status(
        self, 
        job_id: str, 
        status: str, 
        orchestrator_id: str, 
        retries: int = 0,
        base_random_state: int = 42,  # <--- AGGIORNATO: Supporto al seed di failover
        alberi_addestrati: int = 0    # <--- AGGIORNATO: Supporto al progresso del checkpoint
    ) -> None:
        """Aggiorna lo stato di avanzamento tracciando l'orchestratore, i tentativi e i checkpoint."""
        pass

    @abstractmethod
    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        """Imposta lo stato finale su COMPLETED al termine della computazione."""
        pass