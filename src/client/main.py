import json
import os
import sys

from src.shared.config import SystemConfig
from src.shared.factory import get_aws_services
from src.shared.sharedmodels.models import Hyperparameters, InferenceRequest, TrainingRequest
from src.baseline.run_baseline import run_baseline

# 1. Inizializziamo la configurazione leggendo dal file .env
cfg = SystemConfig()

# CENTRALIZZAZIONE: Definiamo il percorso unico per config.json nella cartella mock condivisa
CONFIG_PATH = os.path.join("./.local_storage", "config.json")

BASELINE_CONFIG_PATH = os.path.join("outputs_baseline", "config.json")

# 2. Inizializziamo i servizi globali UNA volta sola all'avvio dello script
try:
    sqs_queue, state_manager = get_aws_services(cfg.env)
except Exception as e:
    print(f"\n[ERRORE] Impossibile inizializzare i servizi per l'ambiente '{cfg.env}': {e}")
    sys.exit(1)


def get_input(prompt: str, default: str = "") -> str:
    user_input = input(prompt).strip()
    return user_input if user_input else default


def load_hyperparameters_from_config(mode:str) -> Hyperparameters:
    if not os.path.exists(BASELINE_CONFIG_PATH):
        raise FileNotFoundError(f"Il file di configurazione '{BASELINE_CONFIG_PATH}' non è stato trovato."  )
    
    with open(BASELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    raw_hp = baseline_data.get("hyperparameters", {})
    if not raw_hp:
        raise ValueError("La sezione 'hyperparameters' è mancante o vuota nel file di configurazione della baseline.")
    known_fields = {"n_estimators", "max_depth", "class_weight", "max_samples", "bootstrap", "tree_type", "target_column"}
    hp_data  = {k:v for k, v in raw_hp.items() if k in known_fields}
    if mode == "federated":
        hp_data["bootstrap"] = False
        hp_data["max_samples"] = 1.0
    hp_data.setdefault("target_column", "Label")
    return Hyperparameters(**hp_data)



def run_predefined_tests():
    """Esegue test predefiniti per verificare il sistema senza input utente."""
    print("\n=== AVVIO TEST DI SISTEMA PREDEFINITI ===")
    print("[INFO] Funzionalità di test automatico globale non ancora implementata.")
    return


def handle_inference():
    print(f"\n=== NUOVO PROCESSO DI INFERENZA ({cfg.mode.upper()}) ===")
    
    # 1. Recupero informazioni di contesto obbligatorie
    job_id = get_input("Inserisci il Job ID del modello addestrato da usare: ")
    if not job_id:
        print("[ERRORE] Il Job ID è obbligatorio.")
        return
        
    data_url = None
    if cfg.mode == "centralized":
        data_url = get_input("Inserisci l'URL/Path dei nuovi dati di test centralizzati: ").strip()
        if not data_url:
            print("[ERRORE] Il percorso dei dati è obbligatorio in modalità centralizzata.")
            return
    else:
        print("[INFO] Modalità Federata: i nodi utilizzeranno le proprie partizioni locali di test trattenute in RAM.")

    # 2. Tentativo di recupero degli iperparametri del modello dal path centralizzato
    hp_obj = None
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
                if saved_config.get("job_id") == job_id or get_input("Usa iperparametri dell'ultimo training locale? (S/N): ", "S").upper() == "S":
                    hp_data = saved_config.get("hyperparameters", {})
                    hp_obj = Hyperparameters(**hp_data)
                    print(f"[INFO] Iperparametri estratti automaticamente (Task rilevato: {hp_obj.tree_type.upper()}).")
        except Exception:
            pass

    # Se non riusciamo a leggerlo, lo chiediamo rapidamente all'utente
    if not hp_obj:
        print("\n[INFO] Impossibile recuperare gli iperparametri in automatico per questo Job ID.")
        tree_type_raw = get_input("Inserisci il tipo di task originale (1 per Classificazione, 2 per Regressione): ", "1")
        tree_type = "classifier" if tree_type_raw == "1" else "regressor"
        hp_obj = Hyperparameters(n_estimators=100, tree_type=tree_type)

    # 3. Validazione tramite il modello Pydantic InferenceRequest
    try:
        inference_request = InferenceRequest(
            job_id=job_id,
            data_url=data_url,
            environment=cfg.env,
            hyperparameters=hp_obj
        )
    except Exception as e:
        print(f"\n [ERRORE VALIDAZIONE STRUTTURA INFERENZA]: {e}")
        return

    # 4. Instradamento sulla coda corretta
    target_queue = "federated_queue" if cfg.mode == "federated" else "centralized_queue"

    try:
        # Inviamo il model_dump() serializzato sulla coda corretta
        sqs_queue.send_message(queue_name=target_queue, message_dict=inference_request.model_dump())
        print(f"\n[OK] Richiesta di inferenza {inference_request.inference_id[:8]} inviata con successo alla coda '{target_queue}'!")
        print(f"[INFO] L'orchestratore riceverà il messaggio e coordinerà i worker via RPC.")
    except Exception as e:
        print(f"[ERRORE] Impossibile inviare la richiesta di inferenza su SQS: {e}")
        
    return


def handle_model_request():
    print("\n=== RICHIESTA E VERIFICA STATO MODELLO ===")
    job_id = get_input("Inserisci il Job ID del modello da verificare: ").strip()
    
    if not job_id:
        print("[ERRORE] Il Job ID è obbligatorio.")
        return

    print(f"[INFO] Interrogazione dello Stato per il Job {job_id[:8]} in corso...")

    try:
        # 1. Interroghiamo lo StateManager per capire lo stato nel cluster locale
        job_status = state_manager.get_job_status(job_id)
        
        if not job_status:
            print(f"\n[ATTENZIONE] Nessun record trovato nel database per il Job ID '{job_id}'.")
            print("[INFO] Verifica che l'ID sia corretto o che l'addestramento sia effettivamente partito.")
            return

        print(f"  • Stato attuale nel Cluster: {job_status.upper()}")

        # 2. Gestione basata sullo stato del ciclo di vita del job
        if job_status.upper() == "QUEUED":
            print(f"\n[IN CODA] Il Job {job_id[:8]} è attualmente in coda su SQS.")
            print("[INFO] Il messaggio è in attesa che l'Orchestratore lo prenda in carico.")
            return

        elif job_status.upper() == "PROCESSING":
            print(f"\n[IN CORSO] Il modello {job_id[:8]} è in fase di addestramento distribuito. ")
            print("[INFO] L'Orchestratore sta coordinando i calcoli paralleli sui nodi Worker via RPC.")
            return

        elif job_status.upper() == "FAILED":
            print(f"\n[FALLITO] L'addestramento per il Job {job_id[:8]} è fallito.")
            print("[INFO] Il sistema di failover ha intercettato un errore infrastrutturale o applicativo.")
            return

        elif job_status.upper() == "COMPLETED":
            print(f"\n[COMPLETATO] L'addestramento per il Job {job_id[:8]} è terminato con successo! ")
            
            # I modelli finali aggregati rimangono nella cartella radice o in './saved_models'
            model_filename = f"model_federated_{job_id}.pkl" if cfg.mode == "federated" else f"model_{job_id}.pkl"
            model_path = os.path.join("./saved_models", model_filename)
            
            # 3. Controllo di persistenza fisica (Solo per ambiente LOCAL)
            if cfg.env == "local":
                if os.path.exists(model_path):
                    print(f"[OK] File binario del modello rilevato in: '{model_path}'")
                    print("[INFO] Il modello è valido e pronto al 100% per ricevere richieste di inferenza.")
                elif os.path.exists(model_filename):
                    print(f"[OK] File binario del modello rilevato nella root: '{model_filename}'")
                else:
                    print(f"[ATTENZIONE] Il DB dichiara 'COMPLETED', ma il file binario '{model_filename}' non è stato trovato.")
            
            elif cfg.env == "aws":
                print(f"[INFO] In ambiente AWS, il file si assume caricato e pronto sul bucket S3.")
                
    except Exception as e:
        print(f"\n[ERRORE] Impossibile recuperare lo stato dal database: {e}")

    return


def handle_training():
    """Gestisce la procedura di richiesta di addestramento basandosi sulla config di boot."""
    print(f"\n=== CONFIGURAZIONE PROCESSO DI ADDESTRAMENTO ({cfg.mode.upper()}) ===")
    
    environment = cfg.env
    mode = cfg.mode

    # 3. SELEZIONE INDIPENDENTE DELLA SORGENTE DATI (Reale vs Sintetico per entrambe le modalità)
    print("\n[3] Selezione della Sorgente Dati:")
    print("  [1] Usa il Dataset REALE (URL S3 pubblico ca-central-1)")
    print("  [2] Genera un dataset SINTETICO per questa esecuzione")
    dataset_choice = get_input("  Scegli l'opzione: ", "1")
    
    if dataset_choice == "2":
        dataset_type = "synthetic"
        if mode == "centralized":
            dataset_path = "/app/data/sintetic_data.csv"
        else:
            dataset_path = "NATIVE_PARTITIONED"
        print(f"  [INFO] Configurato Dataset SINTETICO: {dataset_path}")
    else:
        dataset_type = "real"
        default_s3_url = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
        
        print("  • Inserisci l'URL S3 o il path locale del dataset reale.")
        dataset_path = get_input(f"    (Premi INVIO per il default pubblico): \n    --> ", default_s3_url).strip()
        print(f"  [INFO] Configurato Dataset REALE: {dataset_path}")

    # 4. Configurazione Iperparametri
    print("\n[4] Configurazione Iperparametri da '{BASELINE_CONFIG_PATH}':")
    try:
        hp_obj = load_hyperparameters_from_config(mode)
        print(f"  [OK] Iperparametri caricati da baseline: n_estimators={hp_obj.n_estimators}, max_depth={hp_obj.max_depth}, class_weight={hp_obj.class_weight}, bootstrap={hp_obj.bootstrap}, max_samples={hp_obj.max_samples}, tree_type={hp_obj.tree_type}")
    except FileNotFoundError as e:
        print(f"\n[ERRORE] {e}")
        return
    except Exception as e:
        print(f"\n[ERRORE] Impossibile caricare gli iperparametri dalla baseline: {e}")
        return
        
        
            

    # 5. Validazione Pydantic
    try:
        
        
        request = TrainingRequest(
            environment=environment,
            mode=mode,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            hyperparameters=hp_obj
        )
    except Exception as e:
        print(f"\n [ERRORE VALIDAZIONE STRUTTURA DATI]: {e}")
        return

    # CENTRALIZZAZIONE: Salvataggio file di configurazione locale in .local_storage
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(request.model_dump(), f, indent=2)
        print(f"\n[OK] File di configurazione memorizzato in: '{CONFIG_PATH}'")
    except IOError as e:
        print(f"Impossibile salvare 'config.json' in locale: {e}")

    # 6. Invio del pacchetto e gestione dello stato
    target_queue = "federated_queue" if request.mode == "federated" else "centralized_queue"
    
    try:
        state_manager.initiate_request(job_id=request.job_id, dataset_path=request.dataset_path, seed=request.seed)
        sqs_queue.send_message(queue_name=target_queue, message_dict=request.model_dump())
        print(f"[CLIENT] Richiesta {request.job_id[:8]}... inoltrata con successo alla coda '{target_queue}'!")
        
    except Exception as e:
        print(f"\n [ERRORE INVIO/CODA]: {e}")
        return


def handle_baseline_selection():
    """Interfaccia di instradamento per l'esecuzione della baseline locale."""
    print("\n=== PREPARAZIONE BASELINE LOCALE ===")
    if not os.path.exists(CONFIG_PATH):
        print("[INFO] Nessun file config.json rilevato. Configurazione rapida del dataset per la baseline:")
        print("  [1] Esegui su Dataset Reale")
        print("  [2] Esegui su Dataset Sintetico")
        choice = get_input("  Scelta: ", "1")
        
        dtype = "synthetic" if choice == "2" else "real"
        dummy_config = {"dataset_type": dtype, "hyperparameters": {"n_estimators": 100}}
        
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(dummy_config, f)
            
    run_baseline()


def main():
    while True:
        print("\n=====================================================")
        print("      DISTRIBUTED RANDOM FOREST - CONFIGURATOR       ")
        print(f"      CLUSTER_MODE: {cfg.mode.upper()} | INFRA: {cfg.env.upper()}")
        print("=====================================================\n")

        print("Seleziona la modalità del configuratore:")
        print("[1] Modalità Client Standard")
        print("[2] Modalità Test (Esecuzione test di sistema predefiniti)")
        print("[3] Esci")
        config_mode = get_input("Scelta: ", "1")

        if config_mode == "2":
            run_predefined_tests()
            continue
        elif config_mode == "3":
            print("\nChiusura del Client. Arrivederci!")
            break
        elif config_mode != "1":
            print("\n[ERRORE] Scelta non valida.")
            continue

        print("\n--- MENÙ OPERAZIONI ---")
        print("[1] Avvia processo di addestramento distribuito")
        print("[2] Avvia processo di inferenza distribuito")
        print("[3] Verifica stato modello")
        print("[4] Esegui Baseline Locale")
        print("[5] Torna al menù precedente")
        operation_choice = get_input("Inserisci il numero corrispondente all'operazione: ", "1")         
        
        if operation_choice == "1":
            handle_training()
        elif operation_choice == "2":
            handle_inference()
        elif operation_choice == "3":
            handle_model_request()
        elif operation_choice == "4":
            handle_baseline_selection()
        elif operation_choice == "5":
            continue
        else:
            print("\n[ERRORE] Scelta non valida.")


if __name__ == "__main__":
    main()