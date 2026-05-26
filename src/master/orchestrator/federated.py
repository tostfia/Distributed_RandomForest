import time
from src.shared.factory import get_aws_services

def start_federated_orchestrator(environment: str = "local"):
    print(f"=== ORCHESTRATORE FEDERATO IN ASCOLTO ({environment.upper()}) ===")
    sqs_queue, state_manager = get_aws_services(environment)

    while True:
        # Ascolta SOLO la federated_queue
        sqs_response = sqs_queue.receive_message(queue_name="federated_queue", visibility_timeout=30)

        if sqs_response:
            receipt_handle = sqs_response["ReceiptHandle"]
            payload = sqs_response["Body"]
            job_id = payload["job_id"]

            print(f"\n[Federated Master] Ricevuto Job Federato: {job_id[:8]}...")
            
            state_manager.update_request_status(job_id, "PROCESSING", "Orchestrator-Federated")
            
            # 🚀 LOGICA SPECIALE FEDERATA (Round di addestramento)
            print(f"[Federated Master] Inizio coordinamento dei round federati sui nodi...")
            for round_num in range(1, 4):  # Esempio di 3 round di aggregazione
                print(f"   -> Round {round_num}/3: Raccolta pesi dagli iperparametri {payload['hyperparameters']['n_estimators']} alberi...")
                time.sleep(1)
            
            state_manager.complete_request(job_id, "Orchestrator-Federated")
            sqs_queue.delete_message(receipt_handle)
            print(f"[Federated Master] Modello Globale Federato aggregato con successo.")
            
        time.sleep(1)

if __name__ == "__main__":
    start_federated_orchestrator("local")