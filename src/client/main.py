import json
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from src.testing.engine import TestEngine
from src.shared.config import SystemConfig
from src.shared.factory import get_aws_services
from src.shared.sharedmodels.models import Hyperparameters, InferenceRequest, TrainingRequest
from src.baseline.run_baseline import run_baseline
import shutil

cfg = SystemConfig()

CONFIG_PATH = os.path.join("./.local_storage", "config.json")
HISTORY_PATH = os.path.join("./.local_storage", "requests_history.json")
BASELINE_CONFIG_PATH = os.path.join("outputs_baseline", "config_real.json")

try:
    sqs_queue, state_manager = get_aws_services(cfg.env, role="client")
except Exception as e:
    print(f"\n[ERRORE] Impossibile inizializzare i servizi per l'ambiente '{cfg.env}': {e}")
    sys.exit(1)


def get_input(prompt: str, default: str = "") -> str:
    user_input = input(prompt).strip()
    return user_input if user_input else default


def load_hyperparameters_from_config(mode: str, dataset_type: str = "real") -> Hyperparameters:
    config_path = (
        os.path.join("outputs_baseline", "config_synthetic.json")
        if dataset_type == "synthetic"
        else os.path.join("outputs_baseline", "config_real.json")
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Il file di configurazione '{config_path}' non è stato trovato.")

    with open(config_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    raw_hp = baseline_data.get("hyperparameters", {})
    if not raw_hp:
        raise ValueError("La sezione 'hyperparameters' è mancante o vuota nel file di configurazione della baseline.")

    known_fields = {"n_estimators", "max_depth", "min_samples_split", "class_weight", "max_samples", "bootstrap", "tree_type", "target_column"}
    hp_data = {k: v for k, v in raw_hp.items() if k in known_fields}
    if mode == "federated":
        hp_data["bootstrap"] = False
        hp_data["max_samples"] = 1.0
    hp_data.setdefault("target_column", "Target" if dataset_type == "synthetic" else "Label")
    return Hyperparameters(**hp_data)


def ask_custom_hyperparameters(mode: str, dataset_type: str, tree_type: str) -> Hyperparameters:
    """Chiede all'utente di inserire manualmente gli iperparametri per l'addestramento."""
    print("\n[INFO] Inserimento manuale degli iperparametri (premi INVIO per tenere il default mostrato).")

    n_estimators_raw = get_input("  n_estimators [Default: 100]: ", "100")
    try:
        n_estimators = int(n_estimators_raw)
    except ValueError:
        print(f"  [ATTENZIONE] Valore non valido ('{n_estimators_raw}'), uso il default 100.")
        n_estimators = 100

    max_depth_raw = get_input("  max_depth (vuoto per 'nessun limite') [Default: nessun limite]: ", "")
    max_depth = None
    if max_depth_raw:
        try:
            max_depth = int(max_depth_raw)
        except ValueError:
            print(f"  [ATTENZIONE] Valore non valido ('{max_depth_raw}'), uso 'nessun limite'.")

    min_samples_split_raw = get_input("  min_samples_split (>= 2) [Default: 2]: ", "2")
    try:
        min_samples_split = int(min_samples_split_raw)
        if min_samples_split < 2:
            print(f"  [ATTENZIONE] Valore non valido ('{min_samples_split_raw}'), uso il default 2.")
            min_samples_split = 2
    except ValueError:
        print(f"  [ATTENZIONE] Valore non valido ('{min_samples_split_raw}'), uso il default 2.")
        min_samples_split = 2

    class_weight = None
    if tree_type == "classifier":
        class_weight_raw = get_input(
            "  class_weight ('balanced', 'balanced_subsample' o vuoto) [Default: vuoto]: ", ""
        )
        class_weight = class_weight_raw if class_weight_raw else None

    max_samples_raw = get_input("  max_samples (0 < x <= 1) [Default: 1.0]: ", "1.0")
    try:
        max_samples = float(max_samples_raw)
    except ValueError:
        print(f"  [ATTENZIONE] Valore non valido ('{max_samples_raw}'), uso il default 1.0.")
        max_samples = 1.0

    if mode == "federated":
        print("  [INFO] Modalità FEDERATED: forzo bootstrap=False e max_samples=1.0 per coerenza col protocollo.")
        bootstrap = False
        max_samples = 1.0
    else:
        bootstrap = True

    default_target = "Target" if dataset_type == "synthetic" else "Label"
    target_column = get_input(f"  target_column [Default: {default_target}]: ", default_target)

    return Hyperparameters(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        max_samples=max_samples,
        bootstrap=bootstrap,
        tree_type=tree_type,
        target_column=target_column,
    )


def load_history() -> list:
    """Carica lo storico locale delle richieste inviate (training + inferenza)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError) as e:
        print(f"[ATTENZIONE] Impossibile leggere lo storico locale '{HISTORY_PATH}': {e}")
        return []


def append_history_entry(entry: dict) -> None:
    """Aggiunge una nuova voce allo storico locale delle richieste, senza sovrascrivere le precedenti."""
    history = load_history()
    history.append(entry)
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        print(f"[ATTENZIONE] Impossibile salvare la richiesta nello storico locale: {e}")


def handle_inference():
    print(f"\n=== NUOVO PROCESSO DI INFERENZA ({cfg.mode.upper()}) ===")
    
    # 1. Acquisizione del Job ID e del path dei dati
    job_id = get_input("Inserisci il Job ID del modello addestrato da usare: ").strip()
    if not job_id:
        print("[ERRORE] Il Job ID è obbligatorio.")
        return
        
    if cfg.env == "aws":
        bucket_name = cfg.s3_bucket_name
        default_data_url = f"s3://{bucket_name}/real/"
    else:
        default_data_url = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"

    data_url = get_input(f"Inserisci il path/URL dei dati per l'inferenza [Default: {default_data_url}]: ", default_data_url).strip()

    # 2. Tentativo di recupero degli iperparametri dallo storico locale (append-only),
    # indicizzato per job_id: a differenza di config.json (che viene sovrascritto ad
    # ogni nuovo training o baseline e riflette quindi solo l'ULTIMA esecuzione),
    # lo storico conserva gli iperparametri corretti per ciascun job_id nel tempo.
    hp_obj = None
    matched_training = next(
        (h for h in load_history() if h.get("type") == "training" and h.get("id") == job_id),
        None,
    )
    if matched_training:
        try:
            hp_data = matched_training.get("hyperparameters", {})
            if not hp_data:
                raise ValueError("Voce di storico priva del blocco 'hyperparameters' (job addestrato con una versione precedente del client).")
            hp_obj = Hyperparameters(**hp_data)
            print(f"[INFO] Iperparametri estratti automaticamente dallo storico (Task rilevato: {hp_obj.tree_type.upper()}).")
        except (KeyError, TypeError, ValueError) as e:
            print(f"[INFO] Impossibile ricostruire gli iperparametri dallo storico per il Job ID '{job_id}': {e}")
    else:
        print(f"[INFO] Nessuna voce di training trovata nello storico locale per il Job ID '{job_id}'.")

    # 3. Configurazione manuale di ripiego
    if not hp_obj:
        print("\n[INFO] Impossibile recuperare gli iperparametri in automatico per questo Job ID.")
        tree_type_raw = get_input("Inserisci il tipo di task originale (1 per Classificazione, 2 per Regressione) [Default: 1]: ", "1")
        tree_type = "classifier" if tree_type_raw == "1" else "regressor"
        hp_obj = Hyperparameters(n_estimators=100, tree_type=tree_type)

    # 4. Validazione tramite il modello Pydantic InferenceRequest
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

    # 5. Instradamento sulla coda corretta
    target_queue = "federated_queue.fifo" if cfg.mode == "federated" else "centralized_queue.fifo"

    try:
        sqs_queue.send_message(queue_name=target_queue, message_dict=inference_request.model_dump())
        print(f"\n[OK] Richiesta di inferenza {inference_request.inference_id[:8]} inviata con successo alla coda '{target_queue}'!")
        print(f"[INFO] L'orchestratore riceverà il messaggio e coordinerà i worker via RPC.")

        matched_training = next(
            (h for h in load_history() if h.get("type") == "training" and h.get("id") == job_id),
            None,
        )
        dataset_type = matched_training.get("dataset_type") if matched_training else None

        append_history_entry({
            "type": "inference",
            "id": inference_request.inference_id,
            "job_id": job_id,
            "timestamp": time.time(),
            "environment": inference_request.environment,
            "dataset_type": dataset_type,
            "tree_type": hp_obj.tree_type,
        })
    except Exception as e:
        print(f"[ERRORE] Impossibile inviare la richiesta di inferenza su SQS: {e}")
        
    return


def download_model(job_id: str) -> None:
    """
    Esporta localmente il modello addestrato associato al Job ID.
    In ambiente locale copia il modello dalla cartella saved_models.
    In ambiente AWS scarica il modello dal bucket S3 configurato.
    """
    model_filename = f"model_{job_id}.pkl"

    training_entry = next(
        (
            entry
            for entry in load_history()
            if entry.get("type") == "training"
            and entry.get("id") == job_id
        ),
        None,
    )

    training_mode = (
        training_entry.get("mode")
        if training_entry and training_entry.get("mode")
        else cfg.mode
    )

    print(f"[INFO] Modalità del modello utilizzata per il download: {training_mode.upper()}")

    default_destination = os.path.join("./downloads", model_filename)

    destination_path = get_input(
        f"Percorso di destinazione [Default: {default_destination}]: ",
        default_destination,
    ).strip()

    destination_path = os.path.expanduser(destination_path)
    destination_directory = os.path.dirname(destination_path) or "."
    os.makedirs(destination_directory, exist_ok=True)

    if cfg.env == "local":
        saved_models_path = os.path.join("./saved_models", model_filename)
        root_path = os.path.join(".", model_filename)

        if os.path.exists(saved_models_path):
            source_path = saved_models_path
        elif os.path.exists(root_path):
            source_path = root_path
        else:
            raise FileNotFoundError(
                f"Il modello locale non è stato trovato né in '{saved_models_path}' né in '{root_path}'."
            )

        if os.path.abspath(source_path) == os.path.abspath(destination_path):
            print(f"\n[INFO] Il modello si trova già nel percorso richiesto: '{destination_path}'")
            return

        shutil.copy2(source_path, destination_path)

    elif cfg.env == "aws":
        bucket_name = cfg.s3_bucket_name
        if not bucket_name:
            raise ValueError("DATASETS_BUCKET_NAME non è configurato per l'ambiente AWS.")

        model_key = f"saved_models/{training_mode}/{model_filename}"

        s3_client = boto3.client("s3", region_name=cfg.aws_region)

        try:
            presigned_url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': bucket_name, 'Key': model_key},
                ExpiresIn=3600
            )
            print(f"\n[OK] LINK S3 PER DOWNLOAD DIRETTO VIA BROWSER (valido 1 ora):\n{presigned_url}\n")
        except ClientError as e:
            print(f"[ATTENZIONE] Impossibile generare il Presigned URL: {e}")

        print(f"[INFO] Download da s3://{bucket_name}/{model_key}")

        try:
            s3_client.download_file(bucket_name, model_key, destination_path)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                raise FileNotFoundError(f"Il modello non è presente in s3://{bucket_name}/{model_key}") from exc
            raise

    else:
        raise ValueError(f"Ambiente '{cfg.env}' non supportato.")

    if os.path.exists(destination_path):
        file_size = os.path.getsize(destination_path)
        print(f"\n[OK] Modello scaricato correttamente in: '{destination_path}'")
        print(f"[INFO] Dimensione del file: {file_size} byte")
        print("[ATTENZIONE] I file Pickle devono essere caricati esclusivamente se provengono da fonti attendibili.")
    else:
        print(f"\n[ATTENZIONE] Download terminato ma il file non è stato trovato in '{destination_path}'.")


def handle_model_request():
    print("\n=== RICHIESTA E VERIFICA STATO MODELLO ===")

    job_id = get_input("Inserisci il Job ID del modello da verificare: ").strip()

    if not job_id:
        print("[ERRORE] Il Job ID è obbligatorio.")
        return

    print(f"[INFO] Interrogazione dello Stato per il Job {job_id} in corso...")

    try:
        job_status = state_manager.get_job_status(job_id)

        if not job_status:
            print(f"\n[ATTENZIONE] Nessun record trovato nel database per il Job ID '{job_id}'.")
            print("[INFO] Verifica che l'ID sia corretto o che l'addestramento sia effettivamente partito.")
            return

        print(f"  • Stato attuale nel Cluster: {job_status.upper()}")
        
        if job_status.upper() == "QUEUED":
            print(f"\n[IN CODA] Il Job {job_id} è attualmente in coda su SQS.")
            print("[INFO] Il messaggio è in attesa che l'Orchestratore lo prenda in carico.")
            return

        elif job_status.upper() == "PROCESSING":
            print(f"\n[IN CORSO] Il modello {job_id} è in fase di addestramento distribuito.")
            print("[INFO] L'Orchestratore sta coordinando i calcoli paralleli sui nodi Worker via RPC.")
            return

        elif job_status.upper() == "FAILED":
            print(f"\n[FALLITO] L'addestramento per il Job {job_id} è fallito.")
            print("[INFO] Il sistema di failover ha intercettato un errore infrastrutturale o applicativo.")
            return

        elif job_status.upper() == "COMPLETED":
            print(f"\n[COMPLETATO] L'addestramento per il Job {job_id} è terminato con successo!")

            model_filename = f"model_{job_id}.pkl"
            model_path = os.path.join("./saved_models", model_filename)
            
            if cfg.env == "local":
                if os.path.exists(model_path):
                    print(f"[OK] File binario del modello rilevato in: '{model_path}'")
                    print("[INFO] Il modello è pronto per ricevere richieste di inferenza.")
                elif os.path.exists(model_filename):
                    print(f"[OK] File binario del modello rilevato nella root: '{model_filename}'")
                else:
                    print(f"\n[ATTENZIONE] Il DB dichiara 'COMPLETED', ma il file binario '{model_filename}' non è stato trovato in '{model_path}'.")
                    return

            elif cfg.env == "aws":
                print("[INFO] Il modello risulta completato. La presenza su S3 sarà verificata durante il download.")

            download_choice = get_input("\nVuoi scaricare/esportare il modello? (S/N) [Default: N]: ", "N").strip()

            if download_choice.upper() == "S":
                try:
                    download_model(job_id)
                except Exception as download_error:
                    print(f"\n[ERRORE DOWNLOAD] Impossibile scaricare il modello: {download_error}")

            return

        else:
            print(f"\n[ATTENZIONE] Stato del job non riconosciuto: '{job_status}'.")

    except Exception as e:
        print(f"\n[ERRORE] Impossibile recuperare lo stato dal database: {e}")

    return


def handle_training():
    """Gestisce la procedura di richiesta di addestramento basandosi sulla config di boot."""
    print(f"\n=== CONFIGURAZIONE PROCESSO DI ADDESTRAMENTO ({cfg.mode.upper()}) ===")
    
    environment = cfg.env
    mode = cfg.mode

    # 3. SELEZIONE INDIPENDENTE DELLA SORGENTE DATI
    print("\n[3] Selezione della Sorgente Dati:")
    print("  [1] Usa il Dataset REALE")
    print("  [2] Genera un dataset SINTETICO per questa esecuzione")
    dataset_choice = get_input("  Scegli l'opzione: ", "1")
    
    if dataset_choice == "2":
        dataset_type = "synthetic"
        if mode == "centralized":
            dataset_path = "synthetic/synthetic_dataset.csv"
        else:
            dataset_path = "NATIVE_PARTITIONED"
        print(f"  [INFO] Configurato Dataset SINTETICO: {dataset_path}")
    else:
        dataset_type = "real"
        bucket_name = os.getenv("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket")
        default_s3_url = f"s3://{bucket_name}/real/"
        
        if environment.lower() == "aws":
            prompt_message = f"    Inserisci l'URL S3.\n [Default: {default_s3_url}]: \n    --> "
        else:
            default_s3_url = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"  
            prompt_message = f"     Inserisci il path locale del dataset reale.\n (Premi INVIO per il default): \n    --> "
        
        print("  • Configurazione percorso sorgente dati:")
        dataset_path = get_input(prompt_message, default_s3_url).strip()
        print(f"  [INFO] Configurato Dataset REALE: {dataset_path}")

    # 4. Configurazione Iperparametri
    print(f"\n[4] Configurazione Iperparametri:")
    
    selected_tree_type = "classifier"
    if dataset_type == "synthetic":
        print("  [INFO] Dataset SINTETICO rilevato.")
        print("  Scegli il tipo di esperimento:")
        print("    [1] Classificazione")
        print("    [2] Regressione")
        task_choice = get_input("  Scelta [Default: 1]: ", "1")
        selected_tree_type = "classifier" if task_choice == "1" else "regressor"
    else:
        print("  [INFO] Dataset REALE rilevato (uso default dalla baseline).")

    print("  Vuoi usare la configurazione di DEFAULT della baseline oppure inserire i tuoi iperparametri?")
    print("    [1] Default (baseline)")
    print("    [2] Personalizzati")
    hp_source_choice = get_input("  Scelta [Default: 1]: ", "1")

    try:
        if hp_source_choice == "2":
            hp_obj = ask_custom_hyperparameters(mode, dataset_type, selected_tree_type)
        else:
            hp_obj = load_hyperparameters_from_config(mode, dataset_type)
            hp_obj.tree_type = selected_tree_type
            if selected_tree_type == "regressor":
                hp_obj.class_weight = None

        print(f"  [OK] Task configurato come: {hp_obj.tree_type.upper()}")
        print(f"  [OK] Parametri: n_estimators={hp_obj.n_estimators}, max_depth={hp_obj.max_depth}")

    except Exception as e:
        print(f"\n[ERRORE] Impossibile configurare gli iperparametri: {e}")
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
    target_queue = "federated_queue.fifo" if request.mode == "federated" else "centralized_queue.fifo"
    
    try:
        # In locale scrive direttamente su DB. 
        # In AWS passa per ApiGatewayStateManager (no-op client-side; la scrittura avviene su Lambda)
        state_manager.initiate_request(
            job_id=request.job_id, 
            dataset_path=request.dataset_path, 
            seed=request.seed
        )
        
        # Invia messaggio (Direct SQS in Local / HTTP POST ad API Gateway in AWS)
        sqs_queue.send_message(queue_name=target_queue, message_dict=request.model_dump())
        
        if cfg.env == "aws":
            print(f"[CLIENT] Richiesta {request.job_id[:8]}... inviata con successo via HTTP ad API Gateway!")
        else:
            print(f"[CLIENT] Richiesta {request.job_id[:8]}... inoltrata con successo alla coda '{target_queue}'!")

        append_history_entry({
            "type": "training",
            "id": request.job_id,
            "timestamp": time.time(),
            "mode": request.mode,
            "environment": request.environment,
            "dataset_type": request.dataset_type,
            "tree_type": request.hyperparameters.tree_type,
            "hyperparameters": request.hyperparameters.model_dump(),
        })
        
    except Exception as e:
        print(f"\n [ERRORE INVIO/CODA]: {e}")
        return


def handle_baseline_selection():
    """Interfaccia di instradamento per l'esecuzione della baseline locale."""
    print("\n=== PREPARAZIONE BASELINE LOCALE ===")
    
    print("Seleziona la sorgente dati per il calcolo della baseline:")
    print("  [1] Esegui su Dataset Reale (1% probabilistic sampling)")
    print("  [2] Esegui su Dataset Sintetico (Stress Test)")
    choice = get_input("  Scelta [Default: 1]: ", "1")
    
    dtype = "synthetic" if choice == "2" else "real"
    
    selected_tree_type = "classifier"
    if dtype == "synthetic":
        print("\n  Scegli il tipo di esperimento per la baseline sintetica:")
        print("    [1] Classificazione")
        print("    [2] Regressione")
        task_choice = get_input("  Scelta [Default: 1]: ", "1")
        selected_tree_type = "classifier" if task_choice == "1" else "regressor"

    boot_config = {
        "dataset_type": dtype,
        "tree_type": selected_tree_type
    }
    
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(boot_config, f, indent=2)
    print(f"[OK] Boot configuration registrata: {dtype.upper()} | TASK: {selected_tree_type.upper()}")
        
    run_baseline()


def handle_history_view():
    """Elenca tutte le richieste (training + inferenza) inviate finora, con dataset e stato."""
    print("\n=== STORICO DELLE RICHIESTE INVIATE ===")
    history = load_history()
 
    if not history:
        print("[INFO] Nessuna richiesta trovata nello storico locale.")
        return
 
    dataset_labels = {"real": "REALE", "synthetic": "SINTETICO"}
 
    for i, entry in enumerate(history, start=1):
        entry_type = entry.get("type", "sconosciuto")
        entry_id = entry.get("id", "N/D")
        ts = entry.get("timestamp")
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/D"
        dataset_label = dataset_labels.get(entry.get("dataset_type"), "N/D")
 
        print(f"\n[{i}] {entry_type.upper()} | ID: {entry_id}")
        print(f"    Data: {ts_str}")
        print(f"    Dataset: {dataset_label}")
 
        if entry_type == "training":
            try:
                status = state_manager.get_job_status(entry_id) or "SCONOSCIUTO"
            except Exception as e:
                status = f"ERRORE nel recupero stato ({e})"
            print(f"    Modalità: {entry.get('mode', 'N/D').upper()} | Task: {entry.get('tree_type', 'N/D').upper()}")
            print(f"    Stato: {status}")
        elif entry_type == "inference":
            job_id = entry.get("job_id")
            if job_id:
                print(f"    Job addestrato usato: {job_id}")
            print(f"    Stato: INVIATA (nessun tracciamento di stato disponibile per l'inferenza)")


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
            mode = os.environ.get("SYS_MODE", "centralized")
            env = os.environ.get("SYS_ENV", "local")
            engine = TestEngine(mode=mode, env=env)
            engine.run_scenarios()
            print("\n[INFO] Test Suite completata. Ritorno al menù principale.")
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
        print("[3] Verifica stato modello e download (Pickle)")
        print("[4] Esegui Baseline Locale")
        print("[5] Visualizza storico delle richieste")
        print("[6] Torna al menù precedente")
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
            handle_history_view()
        elif operation_choice == "6":
            continue
        else:
            print("\n[ERRORE] Scelta non valida.")


if __name__ == "__main__":
    main()