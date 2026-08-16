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
    def initiate_request(
        self,
        job_id: str,
        dataset_path: str,
        seed: int,
        hyperparameters: Optional[dict] = None,
        mode: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> None:
        """Registra la richiesta iniziale nel sistema di tracciamento.

        hyperparameters/mode/dataset_type sono persistiti insieme allo stato del job
        così che QUALSIASI client (non solo quello che ha lanciato il training) possa
        recuperarli in seguito tramite get_job_details, ad es. per lanciare un'inferenza
        su un modello addestrato da un altro client.
        """
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
        base_random_state: Optional[int] = None,
        alberi_addestrati: int = 0    
    ) -> None:
        """Aggiorna lo stato di avanzamento tracciando l'orchestratore, i tentativi e i checkpoint."""
        pass

    @abstractmethod
    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        """Imposta lo stato finale su COMPLETED al termine della computazione."""
        pass

    # ------------------------------------------------------------------
    # Tracciamento dei worker task
    # ------------------------------------------------------------------
    @abstractmethod
    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        """Registra che un worker specifico ha ricevuto una parte del lavoro."""
        pass

    @abstractmethod
    def update_worker_task_status(self, job_id: str, worker_id: str, status: str) -> None:
        """Aggiorna lo stato di un worker specifico."""
        pass

    @abstractmethod
    def are_all_workers_done(self, job_id: str, expected_count: int) -> bool:
        """Controlla se tutti i task per un dato Job sono COMPLETED."""
        pass

    @abstractmethod
    def get_active_jobs(self) -> list:
        """Restituisce gli ID di tutti i job attualmente in stato PROCESSING."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Recupera lo stato del job (es. QUEUED, PROCESSING, COMPLETED)."""
        pass

    @abstractmethod
    def get_job_details(self, job_id: str) -> Optional[dict]:
        """Recupera il record completo del job (status, hyperparameters, mode,
        dataset_type, dataset_path, ecc.), non solo lo status.

        Pensato per essere chiamato da un client diverso da quello che ha
        originato il training (scenario multi-client): permette di ricostruire
        gli hyperparameters necessari per lanciare un'inferenza senza dover
        dipendere dallo storico locale di chi ha addestrato il modello.
        Restituisce None se il job non esiste.
        """
        pass

    # ------------------------------------------------------------------
    # Leadership lock globale
    # ------------------------------------------------------------------
    @abstractmethod
    def acquire_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        """Acquisisce un lock distribuito globale."""
        pass

    @abstractmethod
    def refresh_global_lock(self, lock_key: str, owner: str, ttl: int = 30) -> bool:
        """Rinfresca il TTL di un lock distribuito esistente."""
        pass

    @abstractmethod
    def release_global_lock(self, lock_key: str, owner: str) -> bool:
        """Rilascia un lock distribuito globale."""
        pass

    @abstractmethod
    def try_claim_job(self, job_id: str, orchestrator_id: str,lease_seconds: int = 300) -> bool:
        """Tenta di reclamare un job per un orchestratore specifico."""
        pass

    @abstractmethod
    def release_job_lease(self, job_id: str, orchestrator_id: str) -> bool:
        pass