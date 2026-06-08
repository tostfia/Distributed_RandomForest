import time
from typing import Optional
from src.shared.mock_aws.dynamodb import dynamo_db
from src.shared.mock_aws.interfaces import StateManagerInterface

TABLE_NAME = "ModelStatus"

class MockStateManager(StateManagerInterface):
    
    def initiate_request(self, job_id: str, dataset_path: str) -> None:
        """Registra il job appena nato sul database in stato QUEUED."""
        payload = {
            "status": "QUEUED",
            "dataset_path": dataset_path,
            "timestamp": time.time(),
            "retries": 0,
            "last_orchestrator": None,
            "alberi_addestrati": 0,  # Inizializzazione esplicita a 0
            "base_random_state": 42  # Seed di partenza standard
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        print(f"[StateManager] Richiesta registrata (QUEUED) per Job ID: {job_id[:8]}...")

    def obtain_request(self, job_id: str) -> Optional[dict]:
        """Ottieni lo stato attuale del job da DynamoDB."""
        return dynamo_db.get_item(TABLE_NAME, job_id)

    def update_request_status(
        self, 
        job_id: str, 
        status: str, 
        orchestrator_id: str, 
        retries: int = 0, 
        base_random_state: int = 42, 
        alberi_addestrati: int = 0
    ) -> None:
        """Aggiorna lo stato del job tracciando i progressi dell'addestramento e i failover."""
        # Recuperiamo lo stato corrente per non perdere informazioni preesistenti (es: dataset_path)
        current_job = self.obtain_request(job_id) or {}
        
        payload = {
            "status": status,
            "dataset_path": current_job.get("dataset_path"),
            "timestamp": time.time(),
            "retries": retries,
            "last_orchestrator": orchestrator_id,
            "base_random_state": base_random_state,   # Salviamo il seed corrente/aggiornato
            "alberi_addestrati": alberi_addestrati     # Salviamo l'indice del checkpoint degli alberi
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        
        # Log dettagliato per mostrare al professore che lo stato sta avanzando nel DB simulato
        info_progress = f" | Alberi fatti: {alberi_addestrati} | Seed: {base_random_state}" if alberi_addestrati > 0 else ""
        print(f"[StateManager] Job ID: {job_id[:8]}... aggiornato a stato: {status} da {orchestrator_id}{info_progress}")

    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        """Finalizza la richiesta impostando lo stato su COMPLETED."""
        current_job = self.obtain_request(job_id) or {}
        
        payload = {
            "status": "COMPLETED",
            "dataset_path": current_job.get("dataset_path"),
            "timestamp": time.time(),
            "retries": current_job.get("retries", 0),
            "last_orchestrator": orchestrator_id,
            "base_random_state": current_job.get("base_random_state", 42),
            "alberi_addestrati": current_job.get("alberi_addestrati", 0) # Mantiene l'ultimo checkpoint massimo
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        print(f"[StateManager] Job ID: {job_id[:8]}... COMPLETATO con successo da {orchestrator_id}")

    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        """Registra che un worker specifico ha ricevuto una parte del lavoro."""
        # Creiamo una chiave unica: job_id#worker_id
        task_id = f"{job_id}#{worker_id}"
        payload = {
            "status": status,
            "timestamp": time.time(),
            "job_id": job_id,
            "worker_id": worker_id
        }
        # Supponendo che dynamo_db.put_item gestisca anche una tabella 'WorkerTasks'
        dynamo_db.put_item("WorkerTasks", task_id, payload)
        print(f"[StateManager] Task registrato: Job {job_id[:8]} -> Worker {worker_id} in stato {status}")

    def update_worker_task_status(self, job_id: str, worker_id: str, status: str) -> None:
        """Aggiorna lo stato di un worker specifico (chiamato dal worker)."""
        task_id = f"{job_id}#{worker_id}"
        current = dynamo_db.get_item("WorkerTasks", task_id) or {}
        current.update({"status": status, "timestamp": time.time()})
        dynamo_db.put_item("WorkerTasks", task_id, current)
        print(f"[StateManager] Worker {worker_id} ha aggiornato status a {status}")

    def are_all_workers_done(self, job_id: str, expected_count: int) -> bool:
        """Controlla se tutti i task per un dato Job sono COMPLETED."""
        # Qui dovresti interrogare tutti i task che iniziano con job_id#
        # (Se il tuo mock non supporta query complesse, puoi iterare)
        all_tasks = dynamo_db.scan("WorkerTasks") # O metodo equivalente
        job_tasks = [t for t in all_tasks if t.get('job_id') == job_id]
        
        completed_tasks = [t for t in job_tasks if t.get('status') == 'COMPLETED']
        return len(completed_tasks) == expected_count

# Istanza globale esportata per la Factory polimorfa
state_manager = MockStateManager()