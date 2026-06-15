import json
import os
import sys

from src.shared.config import SystemConfig
from src.shared.factory import get_aws_services
from src.shared.sharedmodels.models import Hyperparameters, TrainingRequest
from src.baseline.run_baseline import run_baseline

# 1. Inizializziamo la configurazione leggendo dal file .env
cfg = SystemConfig()

# 2. Inizializziamo i servizi globali UNA volta sola all'avvio dello script
try:
    sqs_queue, state_manager = get_aws_services(cfg.env)
except Exception as e:
    print(f"\n[ERRORE] Impossibile inizializzare i servizi per l'ambiente '{cfg.env}': {e}")
    sys.exit(1)


def get_input(prompt: str, default: str = "") -> str:
    user_input = input(prompt).strip()
    return user_input if user_input else default


def run_predefined_tests():
    """Esegue test predefiniti per verificare il sistema senza input utente."""
    print("\n=== AVVIO TEST DI SISTEMA PREDEFINITI ===")
    print("[INFO] Funzionalità di test automatico globale non ancora implementata.")
    sys.exit(0)


def handle_inference():
    print(f"\n=== NUOVO PROCESSO DI INFERENZA ({cfg.mode.upper()}) ===")
    job_id = get_input("Inserisci il Job ID del modello addestrato da usare: ")
    data_url = get_input("Inserisci l'URL dei nuovi dati da predire: ")
    
    inference_request = {
        "job_id": job_id,
        "data_url": data_url,
        "type": "inference_request"
    }

    try:
        sqs_queue.send_message(queue_name="inference_queue", message_dict=inference_request)
        print(f"[INFO] Richiesta di inferenza inviata per il Job {job_id[:8]}...")
        print("[INFO] Il sistema distribuito sta processando la richiesta e aggregherà i risultati.")
    except Exception as e:
        print(f"[ERRORE] Impossibile inviare la richiesta di inferenza: {e}")
    sys.exit(0)


def handle_model_request():
    print("\n=== RICHIESTA MODELLO ADDESTRATO ===")
    job_id = get_input("Inserisci il Job ID del modello da scaricare: ")
    print(f"[INFO] Download del modello {job_id[:8]} in corso...")
    sys.exit(0)


def handle_training():
    """Gestisce la procedura di richiesta di addestramento basandosi sulla config di boot."""
    print(f"\n=== CONFIGURAZIONE PROCESSO DI ADDESTRAMENTO ({cfg.mode.upper()}) ===")
    
    environment = cfg.env
    mode = cfg.mode

    # 3. Gestione Dinamica del Dataset Path / Sintetico
    dataset_type = "real"  # Default
    if mode == "centralized":
        print("\n[3] Selezione del Dataset:")
        print("  [1] Usa un dataset reale tramite URL/Path")
        print("  [2] Genera un dataset sintetico di test per questa esecuzione")
        dataset_choice = get_input("  Scegli l'opzione: ", "1")
        
        if dataset_choice == "2":
            dataset_path = "/app/data/sintetic_data.csv"
            dataset_type = "synthetic"
            print(f"  [INFO] Utilizzo del dataset sintetico: {dataset_path}")
        else:
            dataset_path = get_input("  • Inserisci l'URL o il path del dataset: ").strip()
            if not dataset_path:
                print("\n[ERRORE] Il path o l'URL del dataset è obbligatorio.")
                sys.exit(1)
    else:
        dataset_path = "NATIVE_PARTITIONED"
        print(f"\n[INFO] Modalità Federata selezionata. I dati si assumono già partizionati sui nodi.")

    # 4. Configurazione Iperparametri
    print("\n[4] Configurazione Matematica degli Alberi:")
    try:
        n_estimators = int(get_input("  • Numero totale di alberi (n_estimators): ", "100"))
        max_depth_raw = get_input("  • Profondità massima (max_depth - Invio per illimitata): ")
        max_depth = int(max_depth_raw) if max_depth_raw else None
        class_weight = get_input("  • Bilanciamento classi (class_weight es: balanced / Invio per None): ") or None
 
        if mode == "centralized":
            bootstrap_raw = get_input("  • Usa bootstrap sampling? (S/N, default S): ", "S").upper()
            bootstrap = bootstrap_raw != "N"
 
            if bootstrap:
                max_samples_raw = get_input("  • Frazione campioni per albero (max_samples, es: 0.8, default 1.0): ", "1.0")
                max_samples = float(max_samples_raw)
                if not (0.0 < max_samples <= 1.0):
                    raise ValueError("max_samples deve essere compreso tra 0 e 1.")
            else:
                max_samples = 1.0
                print("  • max_samples: impostato a 1.0 automaticamente (bootstrap disabilitato)")
        else:
            bootstrap = False
            max_samples = 1.0
            print("  • Bootstrap e max_samples: disabilitati automaticamente (modalità Federata)")
 
        print("  • Tipo di task:")
        print("    [1] Classificazione (usa DecisionTreeClassifier)")
        print("    [2] Regressione     (usa DecisionTreeRegressor)")
        tree_type_raw = get_input("    Scegli: ", "1")
        tree_type = "classifier" if tree_type_raw == "1" else "regressor"
 
        target_column = "Label" 
        print(f"  [INFO] Colonna target impostata automaticamente a: {target_column}")
 
    except ValueError as e:
        print(f"\n[ERRORE] Input non valido: {e}. Riavvia il configuratore.")
        sys.exit(1)

    # 5. Validazione Pydantic
    try:
        hp_obj = Hyperparameters(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            max_samples=max_samples,
            bootstrap=bootstrap,
            tree_type=tree_type,
            target_column=target_column,
        )
        
        request = TrainingRequest(
            environment=environment,
            mode=mode,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            hyperparameters=hp_obj
        )
    except Exception as e:
        print(f"\n [ERRORE VALIDAZIONE STRUTTURA DATI]: {e}")
        sys.exit(1)

    # Salvataggio file di configurazione locale
    try:
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(request.model_dump(), f, indent=2)
        print("\n[OK] File 'config.json' salvato correttamente.")
    except IOError as e:
        print(f"Impossibile salvare 'config.json' in locale: {e}")

    # 6. Invio del pacchetto e gestione dello stato
    target_queue = "federated_queue" if request.mode == "federated" else "centralized_queue"
    
    try:
        state_manager.initiate_request(job_id=request.job_id, dataset_path=request.dataset_path)
        sqs_queue.send_message(queue_name=target_queue, message_dict=request.model_dump())
        print(f"[CLIENT] Richiesta {request.job_id[:8]}... inoltrata con successo alla coda '{target_queue}'!")
        
    except Exception as e:
        print(f"\n [ERRORE INVIO/CODA]: {e}")
        sys.exit(1)


def handle_baseline_selection():
    """Interfaccia di instradamento per l'esecuzione della baseline locale."""
    print("\n=== PREPARAZIONE BASELINE LOCALE ===")
    
    # Se config.json non esiste, creiamo un mini-config al volo per dire alla baseline cosa fare
    if not os.path.exists("config.json"):
        print("[INFO] Nessun file config.json rilevato. Configurazione rapida del dataset per la baseline:")
        print("  [1] Esegui su Dataset Reale")
        print("  [2] Esegui su Dataset Sintetico")
        choice = get_input("  Scelta: ", "1")
        
        dtype = "synthetic" if choice == "2" else "real"
        dummy_config = {"dataset_type": dtype, "hyperparameters": {"n_estimators": 100}}
        
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(dummy_config, f)
            
    # Avvia il codice della baseline che abbiamo corretto nel passaggio precedente
    run_baseline()


def main():
    print("=====================================================")
    print("      DISTRIBUTED RANDOM FOREST - CONFIGURATOR       ")
    print(f"      CLUSTER_MODE: {cfg.mode.upper()} | INFRA: {cfg.env.upper()}")
    print("=====================================================\n")

    print("Seleziona la modalità del configuratore:")
    print("[1] Modalità Client Standard (Interattiva)")
    print("[2] Modalità Test (Esegui test di sistema predefiniti)")
    config_mode = get_input("Scelta: ", "1")

    if config_mode == "2":
        run_predefined_tests()
    elif config_mode != "1":
        print("\n[ERRORE] Scelta non valida. Riavvia il configuratore.")
        sys.exit(1)

    print("\n--- MENÙ OPERAZIONI ---")
    print("[1] Avvia processo di addestramento distribuito")
    print("[2] Avvia processo di inferenza distribuito")
    print("[3] Richiedi modello addestrato")
    print("[4] Esegui Baseline Locale (Singolo Nodo Sequenziale)")
    operation_choice = get_input("Inserisci il numero corrispondente all'operazione: ", "1")         
    
    if operation_choice == "1":
        handle_training()
    elif operation_choice == "2":
        handle_inference()
    elif operation_choice == "3":
        handle_model_request()
    elif operation_choice == "4":
        handle_baseline_selection()
    else:
        print("\n[ERRORE] Scelta non valida. Riavvia il configuratore.")
        sys.exit(1)


if __name__ == "__main__":
    main()