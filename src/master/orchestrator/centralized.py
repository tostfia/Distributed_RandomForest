import json
import os
import time
from src.shared.factory import get_aws_services

def start_centralized_orchestrator(orchestrator_name: str = "Orchestrator-Centralized", environment: str = "local"):
    print(f"=====================================================")
    print(f"  {orchestrator_name.upper()} IN ASCOLTO ({environment.upper()})...")
    print(f"=====================================================\n")

    # Risoluzione Polimorfa delle dipendenze
    sqs_queue, state_manager = get_aws_services(environment)

    while True:
        # Ascolta SOLO la centralized_queue passando i parametri allineati alla nuova interfaccia
        sqs_response = sqs_queue.receive_message(queue_name="centralized_queue", visibility_timeout=30)

        if sqs_response:
            receipt_handle = sqs_response["ReceiptHandle"]  # Token infrastrutturale SQS
            payload = sqs_response["Body"]                  # Payload applicativo
            
            job_id = payload["job_id"]
            mode = payload["mode"]
            dataset_path = payload["dataset_path"]
            hp = payload["hyperparameters"]

            print(f"\n[{orchestrator_name}] Ricevuto Job Centralizzato. SQS ReceiptHandle: {receipt_handle}")
            print(f"[{orchestrator_name}] ID Applicativo (Job ID): {job_id[:8]}...")

            # 1. CONTROLLO DELLO STATO (Logica di recupero/failover richiesta dal professore)
            existing_state = state_manager.obtain_request(job_id)
            
            retries = 0
            base_random_state = 42

            if existing_state:
                current_status = existing_state.get("status")
                retries = existing_state.get("retries", 0)
                
                if current_status == "PROCESSING":
                    # --- CASO FAILOVER ---
                    print(f"[{orchestrator_name}] [FAILOVER DETECTED]: Il precedente Orchestrator ({existing_state.get('last_orchestrator')}) è fallito!")
                    print(f"   Riprendo l'addestramento centralizzato per il dataset '{dataset_path}'.")
                    
                    retries += 1
                    base_random_state = existing_state.get("base_random_state", 42)
                
                elif current_status == "COMPLETED":
                    # Protezione da messaggi fantasma o già elaborati
                    print(f"[{orchestrator_name}] Job già completato in precedenza. Scarto il messaggio ed elimino dalla coda.")
                    sqs_queue.delete_message(receipt_handle)  
                    continue
            else:
                print(f"[{orchestrator_name}] Nuovo Job centralizzato rilevato nel sistema.")

            # 2. CAMBIO STATO: Prenotiamo il job su DynamoDB passando l'ID di questo Orchestrator
            state_manager.update_request_status(
                job_id=job_id, 
                status="PROCESSING", 
                orchestrator_id=orchestrator_name, 
                retries=retries
            )

            # 3. COMPUTAZIONE (Simulazione addestramento Random Forest Centralizzato)
            print(f"[{orchestrator_name}] Addestramento foresta in corso su dataset unico (Alberi: {hp.get('n_estimators')})...")
            time.sleep(15)   

            # 4. COMPLETAMENTO: Aggiorna lo stato a COMPLETED e cancella definitivamente da SQS
            state_manager.complete_request(job_id=job_id, orchestrator_id=orchestrator_name)
            
            # Dimostriamo di avere il lock eliminando il messaggio tramite ReceiptHandle
            sqs_queue.delete_message(receipt_handle)  
            print(f"[{orchestrator_name}] Job {job_id[:8]}... terminato correttamente.")
        
        # Sleep per evitare di sovraccaricare la CPU durante il polling a vuoto
        time.sleep(5)

if __name__ == "__main__":
    # Leggiamo dinamicamente il file config.json per capire l'ambiente configurato dall'utente
    env = "local"
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                env = config.get("environment", "local")
        except Exception:
            pass # Fallback su local se il file è corrotto o mancante
            
    # Avviamo l'orchestratore specializzato
    start_centralized_orchestrator(orchestrator_name="Orchestrator-Centralizzato-A", environment=env)