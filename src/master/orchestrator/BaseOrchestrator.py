from abc import ABC, abstractmethod
import json
import os
import sys
import time
import threading
import boto3
from botocore.exceptions import ClientError
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

    def _get_lock_key(self) -> str:
        return "global_orchestrator_leader_lock"
    
    def _try_acquire_leadership(self,ttl:int = 30) -> bool:
        lock_key = self._get_lock_key()
        
        if self.environment == "local":
            lock_dir = "./.local_storage"
            lock_path = os.path.join(lock_dir, f"{lock_key}.json")
            temp_path = os.path.join(lock_dir, f"{lock_key}.tmp")
            now = time.time()
            
            if os.path.exists(lock_path):
                try:
                    with open(lock_path, "r", encoding="utf-8") as f:
                        lock_data = json.load(f)
                    # Se il lock è ancora valido ed appartiene a un altro, fallisci
                    if lock_data.get("leader") != self.orchestrator_name and (now - lock_data.get("timestamp", 0)) < 25:
                        return False
                except (json.JSONDecodeError, KeyError, ValueError):
                    print(f"[{self.orchestrator_name}] Lock file corrotto o vuoto rilevato sul disco. Tento il ripristino...")
                    pass
                    
                except Exception:
                    pass # File corrotto o rimosso a metà lettura, procedi a sovrascrivere
            
            try:
                os.makedirs(lock_dir, exist_ok=True)
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump({"leader": self.orchestrator_name, "timestamp": now}, f, indent=2)
                os.replace(temp_path, lock_path)  # Atomic replace
                return True
            except:
                return False
        else:
            # Implementazione AWS con Scrittura Condizionale su DynamoDB
            # [STRATEGIA IN AWS]: Sfrutta DynamoDB via state_manager con una scrittura condizionale (Conditional Put)
            try:
                return self.state_manager.acquire_global_lock(lock_key, self.orchestrator_name, ttl=30)
            except AttributeError:
                # Fallback di sicurezza se non ancora mappato nel tuo state_manager AWS custom
                return True
    
    def _refresh_leadership_lock(self):
        """Aggiorna il timestamp del lock per comunicare che il Leader è in salute."""
        lock_key = self._get_lock_key()
        if self.environment == "local":
            lock_dir = "./.local_storage"
            lock_path = os.path.join(lock_dir, f"{lock_key}.json")
            temp_path = os.path.join(lock_dir, f"{lock_key}.tmp")
            try:
                os.makedirs(lock_dir, exist_ok=True)
                # 1. Scriviamo l'aggiornamento sul file temporaneo .tmp
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump({"leader": self.orchestrator_name, "timestamp": time.time()}, f, indent=2)
                # 2. Sostituzione ATOMICA a livello di OS: il file .json non sarà mai vuoto
                os.replace(temp_path, lock_path)
            except Exception as e:
                print(f"[{self.orchestrator_name}] Errore nel rinnovo del lock locale: {e}")
        else:
            try:
                self.state_manager.refresh_global_lock(lock_key, self.orchestrator_name, ttl=30)
            except Exception:
                pass

    def _release_leadership(self):
        """Rilascia il lock globale in fase di spegnimento pulito dell'istanza."""
        lock_key = self._get_lock_key()
        if self.environment == "local":
            lock_path = os.path.join("./.local_storage", f"{lock_key}.json")
            if os.path.exists(lock_path):
                try:
                    with open(lock_path, "r", encoding="utf-8") as f:
                        lock_data = json.load(f)
                    if lock_data.get("leader") == self.orchestrator_name:
                        os.remove(lock_path)
                        print(f"[{self.orchestrator_name}] Lock di leadership globale rilasciato.")
                except Exception:
                    pass
        else:
            try:
                self.state_manager.release_global_lock(lock_key, self.orchestrator_name)
            except Exception:
                pass

    def _heartbeat_loop(self, stop_event: threading.Event, interval: int = 10):
        """Invia heartbeat di rete e tiene in vita il lock di leadership ogni 10 secondi."""
        while not stop_event.is_set():
            try:
                # 1. Aggiorna la presenza nel registro servizi comune
                ServiceRegistry.update_orchestrator_heartbeat(self.orchestrator_name)
                # 2. Rinnova attivamente la scadenza del lock di leadership
                self._refresh_leadership_lock()
            except Exception as e:
                print(f"[{self.orchestrator_name}] Errore durante l'aggiornamento del heartbeat/lock: {e}")

            for _ in range(interval):
                if stop_event.is_set():
                    break
                time.sleep(1)
    def _visibility_heartbeat_loop(self, receipt_handle: str, stop_event: threading.Event, timeout_extension: int = 30, interval: int = 15):
        """
        Invia periodicamente un comando a SQS per estendere l'invisibilità del messaggio
        correntemente in elaborazione, finché l'evento stop_event non viene settato.
        """
        print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT] Thread avviato per il messaggio corrente.")
        while not stop_event.is_set():
            # Attesa interrompibile controllando stop_event ogni secondo
            for _ in range(interval):
                if stop_event.is_set():
                    return
                time.sleep(1)
            
            try:
                # Verifichiamo se il wrapper di SQS espone il metodo per cambiare la visibilità
                if hasattr(self.sqs_queue, "change_message_visibility"):
                    self.sqs_queue.change_message_visibility(
                        queue_name=self.queue_name,
                        receipt_handle=receipt_handle,
                        visibility_timeout=timeout_extension
                    )
                    print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT] Invisibilità del messaggio estesa di altri {timeout_extension}s.")
                else:
                    # Log di debug se ti sei dimenticato di aggiornare il wrapper (Vedi Passo 4)
                    print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT-WARN] Metodo change_message_visibility non trovato sul wrapper.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT-ERROR] Impossibile estendere visibilità SQS: {e}")

    def start(self):
        """Metodo Template: gestisce l'intero ciclo di vita del polling e del failover."""
        print("=====================================================")
        print(f"  {self.orchestrator_name.upper()} IN ASCOLTO ({self.environment.upper()})...")
        print("=====================================================\n")

        ServiceRegistry.register_orchestrator(self.orchestrator_name)

        is_leader = False
        self._stop_heartbeat = threading.Event()
        self.hb_thread = None

        try:
            while True:
                if not is_leader:
                    is_leader = self._try_acquire_leadership()
                    if not is_leader:
                        print(f"[{self.orchestrator_name}] [STANDBY] Un altro orchestratore è Active. In attesa di fault... (Sleep 5s)")
                        time.sleep(5)
                        continue
                    else:
                        print(f"\n[{self.orchestrator_name}] [ACTIVE] !!! LEADERSHIP GLOBALE ACQUISITA !!!")
                        print(f"[{self.orchestrator_name}] Avvio dell'heartbeat thread e attivazione del polling sulla coda: '{self.queue_name}'\n")
                        # Avviamo il thread di heartbeat SOLO dopo aver conquistato la leadership
                        self.hb_thread = threading.Thread(target=self._heartbeat_loop, args=(self._stop_heartbeat,), daemon=True)
                        self.hb_thread.start()

                        #Recupero addestramento 
                        self._perform_active_recovery()
                
                try:
                    sqs_response = self.sqs_queue.receive_message(queue_name=self.queue_name, visibility_timeout=60)

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
            if is_leader:
                self._stop_heartbeat.set()
                if self.hb_thread:
                    self.hb_thread.join(timeout=2)
                self._release_leadership()
            ServiceRegistry.deregister_orchestrator(self.orchestrator_name)
            print(f"[*] Orchestratore rimosso correttamente dalla rete.")

    def _process_job(self, payload: dict, receipt_handle: str):
        """Logica di instradamento del lavoro in base al tipo di richiesta."""
        # 1. Prepariamo e avviamo il thread di Heartbeat per la visibilità SQS
        stop_visibility = threading.Event()
        visibility_thread = threading.Thread(
            target=self._visibility_heartbeat_loop,
            args=(receipt_handle, stop_visibility),
            daemon=True
        )
        visibility_thread.start()
        try:
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
                if receipt_handle:
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
                    alberi_gia_fatti = self._load_checkpoint(job_id, item_data)
                elif current_status == "COMPLETED":
                    print(f"[{self.orchestrator_name}] Job già completato. Scarto.")
                    if receipt_handle:
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
                if receipt_handle:
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
        finally:
            # Segnaliamo al thread di heartbeat di terminare
            stop_visibility.set()
            visibility_thread.join(timeout=2)
            print(f"[{self.orchestrator_name}] [SQS-HEARTBEAT] Thread terminato per il messaggio corrente.")

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
    
    def _perform_active_recovery(self):
        print(f"[{self.orchestrator_name}] Controllo eventuali job in stato PROCESSING per ripristino...")
        job_ids_to_check = []

        if self.environment == "local":
            cp_dir = "./.local_storage/checkpoints"
            if os.path.exists(cp_dir):
                for filename in os.listdir(cp_dir):
                    if filename.startswith("checkpoint_") and filename.endswith(".json"):
                        job_id = filename.replace("checkpoint_", "").replace(".json", "")
                        job_ids_to_check.append(job_id)
        else:
            if hasattr(self.state_manager, "get_active_jobs"):
                job_ids_to_check = self.state_manager.get_active_jobs()
        
        for job_id in job_ids_to_check:
            existing_state = self.state_manager.obtain_request(job_id)
            if existing_state:
                item_data = existing_state.get("Item", existing_state)
                current_status = item_data.get("status")
                old_owner = item_data.get("orchestrator_id")
                if current_status == "PROCESSING" and old_owner != self.orchestrator_name:
                    print(f"[{self.orchestrator_name}] [RECOVERY] Job {job_id[:8]} in stato PROCESSING. Ripristino checkpoint...")
                    print(f"[{self.orchestrator_name}] Il Job {job_id[:8]} era gestito da {old_owner} (mancato).")
                    print(f"[{self.orchestrator_name}] Sincronizzazione stato e subentro immediato in corso...\n")
                    
                    # Ricostruiamo il payload attingendo dallo stato salvato nel DB
                    recovered_payload = {
                        "job_id": job_id,
                        "request_type": item_data.get("request_type", "TRAINING"),
                        "hyperparameters": item_data.get("hyperparameters", item_data.get("hyperparameters", {}))
                    }
                    try:
                        # Eseguiamo il job passando receipt_handle=None perché non proviene da SQS in questo istante
                        self._process_job(recovered_payload, receipt_handle=None)
                    except Exception as e:
                        print(f"[{self.orchestrator_name}] Errore durante il recupero del Job {job_id[:8]}: {e}")
                    