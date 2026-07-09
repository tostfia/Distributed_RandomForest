import time
from typing import Optional
from src.shared.mock_aws.dynamodb.dynamodb import dynamo_db
from src.shared.mock_aws.interfaces import StateManagerInterface

TABLE_NAME = "ModelStatus"

class MockStateManager(StateManagerInterface):
    
    def initiate_request(self, job_id: str, dataset_path: str, seed: int) -> None:
        """Registra il job appena nato sul database in stato QUEUED accettando il seed dal modello Pydantic."""
        payload = {
            "status": "QUEUED",
            "dataset_path": dataset_path,
            "timestamp": time.time(),
            "retries": 0,
            "last_orchestrator": None,
            "alberi_addestrati": 0,
            "base_random_state": seed
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        print(f"[StateManager] Richiesta registrata (QUEUED) per Job ID: {job_id[:8]}... con Seed: {seed}")

    def obtain_request(self, job_id: str) -> Optional[dict]:
        """Ottieni lo stato attuale del job da DynamoDB."""
        return dynamo_db.get_item(TABLE_NAME, job_id)

    def update_request_status(
        self, 
        job_id: str, 
        status: str, 
        orchestrator_id: str, 
        retries: int = 0, 
        base_random_state: Optional[int] = None,
        alberi_addestrati: int = 0
    ) -> None:
        """Aggiorna lo stato del job tracciando i progressi senza imporre un seed di fallback."""
        current_job = self.obtain_request(job_id) or {}
        
        final_seed = base_random_state if base_random_state is not None else current_job.get("base_random_state")
        
        payload = {
            "status": status,
            "dataset_path": current_job.get("dataset_path"),
            "timestamp": time.time(),
            "retries": retries,
            "last_orchestrator": orchestrator_id,
            "base_random_state": final_seed,   
            "alberi_addestrati": alberi_addestrati     
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        
        info_progress = f" | Alberi fatti: {alberi_addestrati} | Seed: {final_seed}" if alberi_addestrati > 0 else f" | Seed: {final_seed}"
        print(f"[StateManager] Job ID: {job_id[:8]}... aggiornato a stato: {status} da {orchestrator_id}{info_progress}")

    def complete_request(self, job_id: str, orchestrator_id: str) -> None:
        """Finalizza la richiesta impostando lo stato su COMPLETED preservando il seed esistente."""
        current_job = self.obtain_request(job_id) or {}
        
        payload = {
            "status": "COMPLETED",
            "dataset_path": current_job.get("dataset_path"),
            "timestamp": time.time(),
            "retries": current_job.get("retries", 0),
            "last_orchestrator": orchestrator_id,
            "base_random_state": current_job.get("base_random_state"),
            "alberi_addestrati": current_job.get("alberi_addestrati", 0) 
        }
        dynamo_db.put_item(TABLE_NAME, job_id, payload)
        print(f"[StateManager] Job ID: {job_id[:8]}... COMPLETATO con successo da {orchestrator_id} | Seed finale: {payload['base_random_state']}")

    def register_worker_task(self, job_id: str, worker_id: str, status: str) -> None:
        """Registra che un worker specifico ha ricevuto una parte del lavoro."""
        task_id = f"{job_id}#{worker_id}"
        payload = {
            "status": status,
            "timestamp": time.time(),
            "job_id": job_id,
            "worker_id": worker_id
        }
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
        response = dynamo_db.scan_table("WorkerTasks") 
        all_tasks = response.get("Items", [])
        
        job_tasks = [t for t in all_tasks if t.get('job_id') == job_id]
        completed_tasks = [t for t in job_tasks if t.get('status') == 'COMPLETED']
        
        print(f"[StateManager] Job {job_id[:8]} -> Worker pronti: {len(completed_tasks)}/{expected_count}")
        return len(completed_tasks) == expected_count
    
    def get_active_jobs(self) -> list:
        """
        Restituisce gli ID di tutti i job attualmente in stato PROCESSING.
        Usato da _perform_active_recovery per individuare job orfani (es. dopo
        un failover dell'orchestratore) e riprenderne il lavoro.
        """
        response = dynamo_db.scan_table(TABLE_NAME)
        all_jobs = response.get("Items", [])
        active_ids = [j.get("job_id") for j in all_jobs if j.get("status") == "PROCESSING" and j.get("job_id")]
        print(f"[StateManager] Scansione job attivi: trovati {len(active_ids)} job in stato PROCESSING.")
        return active_ids

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Recupera lo stato del job (es. QUEUED, PROCESSING, COMPLETED)."""
        response = dynamo_db.get_item(TABLE_NAME, job_id)
        item = response.get("Item") if isinstance(response, dict) and "Item" in response else response
        if item and isinstance(item, dict):
            return item.get("status")
        return None

# Istanza globale esportata per la Factory polimorfa
state_manager = MockStateManager()