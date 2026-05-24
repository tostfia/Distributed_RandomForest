import json

from src.shared.sharedmodels.models import Hyperparameters, TrainingRequest
from src.shared.mock_aws.sqs import sqs_queue

def main():
    print("=====================================================")
    print("      DISTRIBUTED RANDOM FOREST - CONFIGURATOR       ")
    print("=====================================================\n")

    env_choice = input("[1] Ambiente di esecuzione (L - Locale / A - AWS): ").strip().upper()
    environment = "aws" if env_choice == "A" else "local"
    mode_choice = input("[2] Modalità di training (C - Centralizzata / F - Federata): ").strip().upper()
    mode = "federated" if mode_choice == "F" else "centralized"

    dataset_path = input("[3] Inserisci il dataset_path (es: dataset_completo/): ").strip()
    print("\n[4] Configurazione Matematica degli Alberi:")
    n_estimators = int(input("  • Numero totale di alberi (n_estimators): "))
    max_depth_raw = input("  • Profondità massima (max_depth - premi Invio per None): ").strip()
    max_depth = int(max_depth_raw) if max_depth_raw else None
    class_weight = input("  • Bilanciamento classi (class_weight es: balanced): ").strip() or None
    max_samples = float(input("  • Frazione campioni per albero (max_samples es: 0.2): "))

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

    with open("config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    print("\n[OK] File 'config.json' salvato correttamente.")

    # Validazione ed Invio del pacchetto
    hp_obj = Hyperparameters(**config_data["hyperparameters"])
    
    # 3. Creiamo la richiesta: il job_id viene generato automaticamente in background
    request = TrainingRequest(
        environment=config_data["environment"],
        mode=config_data["mode"],
        dataset_path=config_data["dataset_path"],
        hyperparameters=hp_obj
    )

    # Inviamo la richiesta sulla coda
    sqs_queue.send_message(request.model_dump())
    print("[CLIENT] Richiesta di addestramento inoltrata all'Orchestrator.")

if __name__ == "__main__":
    main()