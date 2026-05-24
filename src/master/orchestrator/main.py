import time
from src.shared.mock_aws.sqs import sqs_queue
from src.shared.mock_aws.dynamodb import dynamo_db

def start_orchestrator(orchestrator_name: str = "Orchestrator-Main"):

    while True:
        #Pull dalla coda in comune con un timeout di visibilità
        msg = sqs_queue.receive_message(visibility_timeout=30)

        if msg:
            job_id = msg["job_id"]
            mode = msg["mode"]
            hp = msg["hyperparameters"]

            print(f"\n[{orchestrator_name}] Ricevuta richiesta di addestramento - Job ID: {job_id}")

            #Controllo richiesto: se il job_id è già presente in DynamoDB, significa che è già in lavorazione o completato
            existing_state = dynamo_db.get_item("ModelStaus", job_id)

            if existing_state and existing_state.get("status") == "PROCESSING":
                print(f"[{orchestrator_name}] RECUPERO STATO: Il precedente Orchestrator è fallito!")
                print(f"   Riprendo l'addestramento della foresta '{mode.upper()}' da dove si era interrotto.")
                # Recupera i parametri necessari memorizzati su DynamoDB
                base_random_state = existing_state.get("base_random_state", 42)
            else:
                print(f"[{orchestrator_name}] Nuovo Job rilevato. Inizializzazione stato su DynamoDB.")
                base_random_state = 42
                dynamo_db.put_item("ModelStatus", job_id, {
                    "status": "PROCESSING",
                    "mode": mode,
                    "base_random_state": base_random_state
                })

            # Calcolo fittizio --> poi ovviamente andrà sostituito con l'effettivo addestramento della foresta
            time.sleep(2)   

            dynamo_db.put_item("ModelStatus", job_id, {"status": "COMPLETED", "mode": mode})
            sqs_queue.delete_message(job_id)
        
        time.sleep(1)
