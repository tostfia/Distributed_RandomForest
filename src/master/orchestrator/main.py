import time
from src.shared.mock_aws.sqs import sqs_queue
from src.shared.mock_aws.statemanager import obtain_request, complete_request, update_request_status

def start_orchestrator(orchestrator_name: str = "Orchestrator-Main"):
    print(f"=====================================================")
    print(f"    {orchestrator_name.upper()} IN ASCOLTO...       ")
    print(f"=====================================================\n")

    while True:
        # Polling dalla coda con un timeout di visibilità di 30 secondi
        sqs_response = sqs_queue.receive_message(visibility_timeout=30)

        if sqs_response:
            # -----------------------------------------------------------------
            # MODIFICA: Separazione tra logica infrastrutturale (MOM) e applicativa
            receipt_handle = sqs_response["ReceiptHandle"]  # Token per SQS
            payload = sqs_response["Body"]                  # Dati del messaggio
            
            job_id = payload["job_id"]                      # ID per lo Stato (DynamoDB)
            # -----------------------------------------------------------------
            
            mode = payload["mode"]
            dataset_path = payload["dataset_path"]
            hp = payload["hyperparameters"]

            print(f"\n[{orchestrator_name}] Ricevuto Messaggio. SQS ReceiptHandle: {receipt_handle}")
            print(f"[{orchestrator_name}]  ID Applicativo (Job ID): {job_id[:8]}...")

            # 1. CONTROLLO DELLO STATO (Logica di recupero/failover richiesta dal professore)
            existing_state = obtain_request(job_id)
            
            retries = 0
            
            base_random_state = 42

            if existing_state:
                current_status = existing_state.get("status")
                retries = existing_state.get("retries", 0)
                
                if current_status == "PROCESSING":
                    # --- CASO FAILOVER ---
                    print(f"[{orchestrator_name}] [FAILOVER DETECTED]: Il precedente Orchestrator ({existing_state.get('last_orchestrator')}) è fallito!")
                    print(f"   Riprendo l'addestramento della foresta '{mode.upper()}' per il dataset '{dataset_path}'.")
                    
                    retries += 1
                    base_random_state = existing_state.get("base_random_state", 42)
                
                elif current_status == "COMPLETED":
                    # Protezione da messaggi fantasma o già elaborati
                    print(f"[{orchestrator_name}] Job già completato in precedenza. Scarto il messaggio.")
                    sqs_queue.delete_message(receipt_handle)  # Usiamo il ReceiptHandle
                    continue
            else:
                print(f"[{orchestrator_name}] Nuovo Job rilevato nel sistema.")

            # 2. CAMBIO STATO: Prenotiamo il job su DynamoDB passando l'ID di questo Orchestrator
            update_request_status(
                job_id=job_id, 
                status="PROCESSING", 
                orchestrator_id=orchestrator_name, 
                retries=retries
            )

            # 3. COMPUTAZIONE (Sostituire in futuro con l'addestramento Distributed Random Forest)
            print(f"[{orchestrator_name}] Addestramento in corso (Modo: {mode}, Alberi: {hp.get('n_estimators')})...")
            time.sleep(2)   

            # 4. COMPLETAMENTO: Aggiorna lo stato a COMPLETED e cancella definitivamente da SQS
            complete_request(job_id=job_id, orchestrator_id=orchestrator_name)
            
            # Usiamo il ReceiptHandle per cancellare, l'Orchestrator dimostra di avere il "lock" sul messaggio
            sqs_queue.delete_message(receipt_handle)  
            print(f"[{orchestrator_name}] Job {job_id[:8]}... terminato correttamente.")
        
        # Piccolo sleep per evitare di sovraccaricare la CPU durante il polling a vuoto
        time.sleep(1)

if __name__ == "__main__":
    start_orchestrator("Orchestrator-Nodo-A")