from abc import ABC, abstractmethod
import json
import os
import sys
import time
import threading

from src.shared.config import SystemConfig
from src.shared.factory import get_aws_services, DatasetDAOFactory
from src.shared.binding.serviceregistry import ServiceRegistry


class BaseOrchestrator(ABC):
    def __init__(self, orchestrator_name: str, queue_name: str):
        # 1. Leggiamo l'ambiente direttamente dal file .env tramite SystemConfig
        self.cfg = SystemConfig()
        self.environment = self.cfg.env
        
        self.orchestrator_name = orchestrator_name
        self.queue_name = queue_name
        
        try:
            self.sqs_queue, self.state_manager = get_aws_services(self.environment)
        except Exception as e:
            print(f"[{self.orchestrator_name.upper()}] Errore inizializzazione servizi: {e}")
            sys.exit(1)

    def _heartbeat_loop(self, stop_event: threading.Event, interval: int = 30):
        while not stop_event.is_set():
            try:
                ServiceRegistry.update_orchestrator_heartbeat(self.orchestrator_name)
            except Exception as e:
                print(f"[{self.orchestrator_name}] Errore durante l'aggiornamento del heartbeat: {e}")

            for _ in range(interval):
                if stop_event.is_set():
                    break
                time.sleep(1)

    def start(self):
        """Metodo Template: gestisce l'intero ciclo di vita del polling e del failover."""
        print("=====================================================")
        print(f"  {self.orchestrator_name.upper()} IN ASCOLTO ({self.environment.upper()})...")
        print("=====================================================\n")

        ServiceRegistry.register_orchestrator(self.orchestrator_name)

        self._stop_heartbeat = threading.Event()
        self.hb_thread = threading.Thread(target=self._heartbeat_loop, args=(self._stop_heartbeat,), daemon=True)
        self.hb_thread.start()

        try:
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
                    import traceback          
                    traceback.print_exc()
                    time.sleep(10)
        except KeyboardInterrupt:
            print(f"\n[-] Interruzione manuale intercettata sull'orchestrattore {self.orchestrator_name}")
        finally:
            print(f"[*] Chiusura dei servizi in corso per {self.orchestrator_name}...")
            self._stop_heartbeat.set()
            self.hb_thread.join(timeout=1)
            ServiceRegistry.deregister_orchestrator(self.orchestrator_name)
            print(f"[*] Orchestratore rimosso correttamente dalla rete.")

    def _process_job(self, payload: dict, receipt_handle: str):
        """Logica di instradamento del lavoro in base al tipo di richiesta."""
        job_id = payload.get("job_id")
        # Estraiamo il tipo di richiesta dal payload, di default assumiamo sia "TRAINING" per retrocompatibilità
        request_type = payload.get("request_type", "TRAINING").upper()

        if request_type == "INFERENCE":
            print(f"\n[{self.orchestrator_name}] Ricevuta richiesta di INFERENZA per il Job ID: {job_id[:8]}...")
            try:
                # Eseguiamo la predizione distribuita (implementata dalle classi figlie)
                self._execute_inference_step(payload)
                # Eliminiamo il messaggio solo a successo ottenuto
                self.sqs_queue.delete_message(receipt_handle)
                print(f"[{self.orchestrator_name}] Inferenza per Job {job_id[:8]} completata con successo.")
            except Exception as inf_error:
                print(f"[{self.orchestrator_name}] [ERRORE DURANTE INFERENZA]: {inf_error}")
                import traceback
                traceback.print_exc()
            return

        # --- SE NON È INFERENZA, GESTIAMO IL CORRETTO FLUSSO DI TRAINING (VECCHIA LOGICA) ---
        status = self.state_manager.get_job_status(job_id)
        if status == "COMPLETED":
            print(f"[INFO] Job {job_id[:8]} già completato. Ignoro messaggio duplicato.")
            self.sqs_queue.delete_message(receipt_handle)
            return 
            
        hp = payload.get("hyperparameters", {})
        existing_state = self.state_manager.obtain_request(job_id)
        retries = 0
        base_random_state = 123
        alberi_gia_fatti = 0

        if existing_state:
            item_data = existing_state.get("Item", existing_state)
            current_status = item_data.get("status")
            retries = item_data.get("retries", 0)
            base_random_state = item_data.get("base_random_state", 123)
            
            if current_status == "PROCESSING":
                print(f"[{self.orchestrator_name}] [FAILOVER DETECTED] Riprendo il lavoro del nodo fallito.")
                retries += 1
                base_random_state += retries
                alberi_gia_fatti = self._load_checkpoint(job_id, item_data)
            elif current_status == "COMPLETED":
                print(f"[{self.orchestrator_name}] Job già completato. Scarto.")
                self.sqs_queue.delete_message(receipt_handle)
                return
        
        self.state_manager.update_request_status(
            job_id=job_id, 
            status="PROCESSING", 
            orchestrator_id=self.orchestrator_name, 
            retries=retries,
            base_random_state=base_random_state,
            alberi_addestrati=alberi_gia_fatti
        )

        start_dist = time.perf_counter()

        try:
            alberi_totali = hp.get("n_estimators", 100)
            # Definiamo un numero di alberi non fisso 
            num_worker_attuali  = max(1, self._get_active_worker_count())
            step_alberi = max(20, num_worker_attuali * 10)  # Step dinamico basato sul numero di worker attivi
            current_alberi = alberi_gia_fatti

            while current_alberi < alberi_totali:
                prossimo_target = min(current_alberi + step_alberi, alberi_totali)
                successo = self._execute_training_step(payload, current_alberi, prossimo_target, base_random_state)

                if not successo:
                    print(f"[{self.orchestrator_name}] Risorse insufficienti per Job {job_id[:8]}. In attesa...")
                    return
                
                current_alberi = prossimo_target
                self._save_checkpoint(job_id, current_alberi, retries, base_random_state)
        
            t_dist = time.perf_counter() - start_dist 
            self.state_manager.complete_request(job_id=job_id, orchestrator_id=self.orchestrator_name)
            self.sqs_queue.delete_message(receipt_handle)
            self._clean_checkpoint(job_id)

            self._generate_performance_report(job_id, t_dist)
            print(f"[{self.orchestrator_name}] Job {job_id[:8]} completato con successo.")

        except Exception as eval_error:
            print(f"[{self.orchestrator_name}] [ERRORE APPLICATIVO]: {eval_error}")
            import traceback
            traceback.print_exc()
            self.state_manager.update_request_status(
                job_id=job_id, 
                status="FAILED", 
                orchestrator_id="SYSTEM_ERR",
                retries=retries,
                base_random_state=base_random_state,
                alberi_addestrati=current_alberi
            )

    @abstractmethod
    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        pass

    @abstractmethod
    def _execute_inference_step(self, payload: dict):
        pass

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, seed: int):
        checkpoint_data = {"alberi_addestrati": current_alberi}
        
        if self.environment == "local":
            local_cp_dir = os.path.join("./.local_storage", "checkpoints")
            os.makedirs(local_cp_dir, exist_ok=True)
            
            cp_path = os.path.join(local_cp_dir, f"checkpoint_{job_id}.json")
            with open(cp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
        else:
            try:
                import pandas as pd
                dao = DatasetDAOFactory.get_dao(self.environment)
                df_cp = pd.DataFrame([checkpoint_data])
                dao.save_dataset(df_cp, f"s3://my-cluster-checkpoints-bucket/checkpoint_{job_id}.csv")
                print(f"   [S3-CHECKPOINT] Salvato checkpoint ({current_alberi} alberi) su S3.")
            except Exception as e:
                print(f"   [S3-CHECKPOINT-ERROR] Impossibile salvare il checkpoint su S3: {e}")
        
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
            path = os.path.join("./.local_storage", "checkpoints", f"checkpoint_{job_id}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f).get("alberi_addestrati", db_val)
        else:
            return db_val
        return db_val

    def _clean_checkpoint(self, job_id: str):
        if self.environment == "local":
            path = os.path.join("./.local_storage", "checkpoints", f"checkpoint_{job_id}.json")
            if os.path.exists(path):
                os.remove(path)
    
    def _generate_performance_report(self, job_id: str, t_dist: float):
        print("\n" + "═" * 75)
        print(f"  REPORT PRESTAZIONALE DISTRIBUITO - JOB {job_id[:8]}")
        print("═" * 75)
        print(f"  Tempo totale addestramento (T_dist):   {t_dist:.4f} s")
        print(f"  Worker utilizzati:                     {self._get_active_worker_count()}")
        print("═" * 75 + "\n")
    
    def _get_active_worker_count(self):
        workers = ServiceRegistry.get_available_workers(self.environment)
        return len(workers)