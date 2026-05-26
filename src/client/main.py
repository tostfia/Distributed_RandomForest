import json
import os
import sys

from src.shared.factory import get_aws_services
from src.shared.sharedmodels.models import Hyperparameters, TrainingRequest

def get_input(prompt: str, default: str = "") -> str:
    user_input = input(prompt).strip()
    return user_input if user_input else default

def main():
    print("=====================================================")
    print("      DISTRIBUTED RANDOM FOREST - CONFIGURATOR       ")
    print("=====================================================\n")

    # 1. Scelta dell'Ambiente
    env_choice = get_input("[1] Ambiente di esecuzione (L - Locale / A - AWS): ").strip().upper()
    environment = "aws" if env_choice == "A" else "local"
    
    # 2. Scelta della Modalità
    mode_choice = get_input("[2] Modalità di training (C - Centralizzata / F - Federata): ").strip().upper()
    mode = "federated" if mode_choice == "F" else "centralized"

    # 3. Gestione Dinamica del Dataset Path
    if mode == "centralized":
        dataset_path = get_input("[3] Inserisci il dataset_path (es: dataset_completo/): ").strip()
        if environment == "local" and not os.path.exists(dataset_path):
            print(f" [ATTENZIONE] Il path locale '{dataset_path}' non sembra esistere. Proseguo comunque...")
    else:
        dataset_path = "NATIVE_PARTITIONED"
        print(f" [INFO] Modalità Federata selezionata. I dati si assumono già partizionati sui nodi.")

    # 4. Configurazione Iperparametri
    print("\n[4] Configurazione Matematica degli Alberi:")
    try:
        n_estimators = int(get_input("  • Numero totale di alberi (n_estimators): ", "100"))
        max_depth_raw = get_input("  • Profondità massima (max_depth - Invio per illimitata): ")
        max_depth = int(max_depth_raw) if max_depth_raw else None
        class_weight = get_input("  • Bilanciamento classi (class_weight es: balanced / Invio per None): ") or None
        max_samples_raw = get_input("  • Frazione campioni per albero (max_samples): ", "1.0")
        max_samples = float(max_samples_raw)
        
        if not (0.0 < max_samples <= 1.0):
            raise ValueError("max_samples deve essere compreso tra 0 e 1.")
            
    except ValueError as e:
        print(f"\n[ERRORE] Input non valido: {e}. Riavvia il configuratore.")
        sys.exit(1)

    # 5. Inizializzazione Servizi (Spostata e protetta da try-except)
    try:
        sqs_queue, state_manager = get_aws_services(environment)
    except Exception as e:
        print(f"\n[ERRORE] Impossibile inizializzare i servizi per l'ambiente '{environment}': {e}")
        sys.exit(1)

    # 6. Validazione Pydantic
    try:
        hp_obj = Hyperparameters(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            max_samples=max_samples
        )
        
        request = TrainingRequest(
            environment=environment,
            mode=mode,
            dataset_path=dataset_path,
            hyperparameters=hp_obj
        )
    except Exception as e:
        print(f"\n [ERRORE VALIDAZIONE STRUTTURA DANI]: {e}")
        sys.exit(1)

    # Salvataggio file di configurazione locale (Ora include TUTTI i dati reali inclusi i default di Pydantic)
    try:
        with open("config.json", "w", encoding="utf-8") as f:
            # Esportiamo direttamente il modello Pydantic, così config.json include il Job ID generato!
            json.dump(request.model_dump(), f, indent=2)
        print("\n[OK] File 'config.json' salvato correttamente.")
    except IOError as e:
        print(f"Impossibile salvare 'config.json' in locale: {e}")

    # 7. Invio del pacchetto e gestione dello stato
    target_queue = "federated_queue" if request.mode == "federated" else "centralized_queue"
    
    try:
        # 1. Registriamo lo stato iniziale su DynamoDB
        state_manager.initiate_request(job_id=request.job_id, dataset_path=request.dataset_path)
        
        # 2. Inoltriamo il payload alla coda
        sqs_queue.send_message(queue_name=target_queue, message_dict=request.model_dump())
        print(f"[CLIENT] Richiesta {request.job_id[:8]}... inoltrata con successo alla coda '{target_queue}'!")
        
    except Exception as e:
        print(f"\n [ERRORE INVIO/CODA]: {e}")
        print(" [ATTENZIONE] La richiesta potrebbe essere registrata su DB ma non inviata alla coda.")
        sys.exit(1)

if __name__ == "__main__":
    main()