from abc import ABC, abstractmethod
import json
import os
import sys
import time
from src.shared.factory import get_aws_services

class BaseOrchestrator(ABC):
    def __init__(self, orchestrator_name: str, queue_name: str, environment: str = "local"):
        self.orchestrator_name = orchestrator_name
        self.queue_name = queue_name
        self.environment = environment
        
        try:
            self.sqs_queue, self.state_manager = get_aws_services(self.environment)
        except Exception as e:
            print(f"[{self.orchestrator_name.upper()}] Errore inizializzazione servizi: {e}")
            sys.exit(1)

    def start(self):
        """Metodo Template: gestisce l'intero ciclo di vita del polling e del failover."""
        print("=====================================================")
        print(f"  {self.orchestrator_name.upper()} IN ASCOLTO ({self.environment.upper()})...")
        print("=====================================================\n")

        while True:
            try:
                sqs_response = self.sqs_queue.receive_message(queue_name=self.queue_name, visibility_timeout=300)

                if not sqs_response:
                    time.sleep(5)
                    continue

                receipt_handle = sqs_response["ReceiptHandle"]
                payload = sqs_response["Body"]
                
                self._process_job(payload, receipt_handle)

            except Exception as infra_error:
                print(f"\n[{self.orchestrator_name}] [ERRORE INFRASTRUTTURALE]: {infra_error}")
                time.sleep(10)

    def _process_job(self, payload: dict, receipt_handle: str):
        """Logica di gestione dello stato e orchestrazione del Job."""
        job_id = payload["job_id"]
        hp = payload["hyperparameters"]
        
        print(f"\n[{self.orchestrator_name}] Ricevuto Job. ID: {job_id[:8]}...")

        existing_state = self.state_manager.obtain_request(job_id)
        retries = 0
        base_random_state = 42
        alberi_gia_fatti = 0

        if existing_state:
            current_status = existing_state.get("status")
            retries = existing_state.get("retries", 0)
            base_random_state = existing_state.get("base_random_state", 42)
            
            if current_status == "PROCESSING":
                print(f"[{self.orchestrator_name}] [FAILOVER DETECTED] Riprendo il lavoro del nodo fallito.")
                retries += 1
                base_random_state += retries
                alberi_gia_fatti = self._load_checkpoint(job_id, existing_state)
            elif current_status == "COMPLETED":
                print(f"[{self.orchestrator_name}] Job già completato. Scarto.")
                self.sqs_queue.delete_message(receipt_handle)
                return
        
        # Primo blocco del job su DB (senza avanzamento alberi iniziale)
        self.state_manager.update_request_status(
            job_id=job_id, 
            status="PROCESSING", 
            orchestrator_id=self.orchestrator_name, 
            retries=retries,
            base_random_state=base_random_state,
            alberi_addestrati=alberi_gia_fatti
        )

        try:
            alberi_totali = hp.get("n_estimators", 100)
            step_alberi = 20
            current_alberi = alberi_gia_fatti

            while current_alberi < alberi_totali:
                prossimo_target = min(current_alberi + step_alberi, alberi_totali)
                
                # CHIAMATA AL METODO ASTRATTO (Passiamo anche il seed, fondamentale!)
                self._execute_training_step(payload, current_alberi, prossimo_target, base_random_state)
                
                current_alberi = prossimo_target
                # Salvataggio del checkpoint integrato con lo StateManager aggiornato
                self._save_checkpoint(job_id, current_alberi, retries, base_random_state)

            self.state_manager.complete_request(job_id=job_id, orchestrator_id=self.orchestrator_name)
            self.sqs_queue.delete_message(receipt_handle)
            self._clean_checkpoint(job_id)
            print(f"[{self.orchestrator_name}] Job {job_id[:8]} completato con successo.")

        except Exception as eval_error:
            print(f"[{self.orchestrator_name}] [ERRORE APPLICATIVO]: {eval_error}")
            self.state_manager.update_request_status(
                job_id=job_id, 
                status="PROCESSING", 
                orchestrator_id="SYSTEM_ERR",
                retries=retries,
                base_random_state=base_random_state,
                alberi_addestrati=current_alberi
            )

    @abstractmethod
    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        """Contiene la vera logica di addestramento (Centralizzata o Federata)."""
        pass

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, seed: int):
        if self.environment == "local":
            os.makedirs("checkpoints", exist_ok=True)
            with open(f"checkpoints/checkpoint_{job_id}.json", "w") as f:
                json.dump({"alberi_addestrati": current_alberi}, f)
        else:
            print(f"   [S3-CHECKPOINT] Inviato checkpoint ({current_alberi} alberi) su S3.")
        
        # Chiamata corretta al metodo unificato di DynamoDB/Mock
        self.state_manager.update_request_status(
            job_id=job_id,
            status="PROCESSING",
            orchestrator_id=self.orchestrator_name,
            retries=retries,
            base_random_state=seed,
            alberi_addestrati=current_alberi
        )

    def _load_checkpoint(self, job_id: str, existing_state: dict) -> int:
        db_val = existing_state.get("alberi_addestrati", 0)
        if self.environment == "local":
            path = f"checkpoints/checkpoint_{job_id}.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f).get("alberi_addestrati", db_val)
        return db_val

    def _clean_checkpoint(self, job_id: str):
        if self.environment == "local":
            path = f"checkpoints/checkpoint_{job_id}.json"
            if os.path.exists(path):
                os.remove(path)