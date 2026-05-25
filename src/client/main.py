import json
import os
import sys

from src.shared.sharedmodels.models import Hyperparameters, TrainingRequest
from src.shared.mock_aws.sqs import sqs_queue

def get_input(prompt: str, default: str = "") -> str:
    user_input = input(prompt).strip()
    return user_input if user_input else default

def main():
    print("=====================================================")
    print("      DISTRIBUTED RANDOM FOREST - CONFIGURATOR       ")
    print("=====================================================\n")

    env_choice = get_input("[1] Ambiente di esecuzione (L - Locale / A - AWS): ").strip().upper()
    environment = "aws" if env_choice == "A" else "local"
    
    mode_choice = get_input("[2] Modalità di training (C - Centralizzata / F - Federata): ").strip().upper()
    mode = "federated" if mode_choice == "F" else "centralized"

    dataset_path = get_input("[3] Inserisci il dataset_path (es: dataset_completo/): ").strip()
    print("\n[4] Configurazione Matematica degli Alberi:")
    if environment == "local" and not os.path.exists(dataset_path):
        print(f"[ATTENZIONE] Il path locale '{dataset_path}' non sembra esistere. Proseguo comunque...")

    # Configurazione Iperparametri con gestione degli errori
    print("\n[4] Configurazione Matematica degli Alberi:")
    try:
        n_estimators = int(get_input("  • Numero totale di alberi (n_estimators) [100]: ", "100"))
        
        max_depth_raw = get_input("  • Profondità massima (max_depth - Invio per illimitata): ")
        max_depth = int(max_depth_raw) if max_depth_raw else None
        
        class_weight = get_input("  • Bilanciamento classi (class_weight es: balanced / Invio per None): ") or None
        
        max_samples_raw = get_input("  • Frazione campioni per albero (max_samples) [1.0]: ", "1.0")
        max_samples = float(max_samples_raw)
        if not (0.0 < max_samples <= 1.0):
            raise ValueError("max_samples deve essere compreso tra 0 e 1.")
            
    except ValueError as e:
        print(f"\n[ERRORE] Input non valido: {e}. Riavvia il configuratore.")
        sys.exit(1)

    # Struttura del file JSON generata dall'utente
    config_data = {
        "environment": environment,
        "mode": mode,
        "dataset_path": dataset_path,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
            "max_samples": max_samples
        }
    }

   # Salvataggio file di configurazione
    try:
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print("\n[OK] File 'config.json' salvato correttamente.")
    except IOError as e:
        print(f"Impossibile salvare 'config.json' in locale: {e}")

    # Validazione ed Invio del pacchetto
    try:
        hp_obj = Hyperparameters(**config_data["hyperparameters"])
        
        request = TrainingRequest(
            environment=config_data["environment"],
            mode=config_data["mode"],
            dataset_path=config_data["dataset_path"],
            hyperparameters=hp_obj
        )

        sqs_queue.send_message(request.model_dump())
        print("[CLIENT] Richiesta di addestramento inoltrata correttamente all'Orchestrator.")
        
    except Exception as e:
        print(f"\n [ERRORE VALIDAZIONE/INVIO]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()