"""
Storage condiviso per i blob di alberi serializzati prodotti dai worker
durante l'addestramento distribuito (locale: file su './.local_storage',
AWS: oggetti S3). Estratto da BaseWorker.py perché ora anche l'Orchestratore
deve poter rileggere questi blob direttamente, invece di riceverli come
valore di ritorno RPC (vedi hang osservato nello Scenario 2 di scalabilità
su payload >1GB: RPyC non è pensato per trasferire blob di queste dimensioni
come valore di ritorno di una chiamata sincrona).

Destinazione: src/shared/utilities/task_storage.py
"""
import os
import json
import pickle
import boto3
from botocore.exceptions import ClientError


def _derive_job_id(source_info: str) -> str:
    """
    Estrazione del job_id da source_info, unica fonte di verità condivisa da
    tutte le funzioni di questo modulo (path monolitico, manifest, parti),
    per evitare che la stessa logica venga duplicata e potenzialmente
    disallineata in più punti (come accadeva prima tra BaseWorker e questo
    modulo per lo schema monolitico).
    """
    filename = os.path.basename(source_info)  # es: shared_train_12345.csv
    return filename.replace("shared_train_", "").replace(".csv", "")


def get_task_storage_paths(source_info: str, base_seed: int, num_trees: int,
                            bucket_env_var: str = "DATASETS_BUCKET_NAME",
                            default_bucket: str = "my-cluster-datasets-bucket-759804778194-us-east-1-an"):
    """
    Genera i percorsi per lo storage condiviso basandosi sul TASK.
    Estrae il job_id dal source_info per evitare collisioni tra job diversi.
    Identico a BaseWorker._get_task_storage_paths: unica fonte di verità
    per come viene calcolata la chiave, usata sia da chi scrive (worker)
    sia da chi rilegge (orchestratore).
    """
    job_id = _derive_job_id(source_info)

    local_dir = os.path.join("./.local_storage", "trained_tasks")
    base_name = f"task_{job_id}_seed_{base_seed}_trees_{num_trees}"
    local_meta_path = os.path.join(local_dir, base_name + ".meta.json")
    local_bin_path = os.path.join(local_dir, base_name + ".bin")

    s3_bucket = os.environ.get(bucket_env_var, default_bucket)
    s3_key = f"tasks/{job_id}/task_seed_{base_seed}_trees_{num_trees}.pkl"

    return local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key


def get_task_part_key(source_info: str, base_seed: int, num_trees: int, part_idx: int) -> str:
    """
    Chiave (path relativo, usabile sia in locale sia su S3 tramite
    save/load_bytes_to_shared_storage) di UNA SINGOLA PARTE/batch di un task.

    Ogni parte contiene solo gli alberi del proprio batch, non l'intero
    chunk: questo è ciò che permette al worker di liberare la memoria di un
    batch subito dopo averlo scritto, invece di accumulare tutti gli alberi
    del task in RAM fino alla fine (vedi manifest per la ricomposizione).
    """
    job_id = _derive_job_id(source_info)
    return f"tasks/{job_id}/task_seed_{base_seed}_trees_{num_trees}_part_{part_idx}.pkl"


def get_task_manifest_key(source_info: str, base_seed: int, num_trees: int) -> str:
    """
    Chiave del manifest JSON che elenca le parti scritte per un task.
    È il file scritto per ULTIMO, dopo tutte le parti: la sua presenza
    segnala che il task è completo (vedi load_task_parts_from_shared_storage),
    stesso principio dell'ack RPC che oggi segnala il completamento
    all'Orchestratore.
    """
    job_id = _derive_job_id(source_info)
    return f"tasks/{job_id}/task_seed_{base_seed}_trees_{num_trees}.manifest.json"


def save_bytes_to_shared_storage(key: str, data: bytes, environment: str, node_name: str = "",
                                  bucket_env_var: str = "DATASETS_BUCKET_NAME",
                                  default_bucket: str = "my-cluster-datasets-bucket-759804778194-us-east-1-an"):
    """
    Salva byte grezzi sotto una chiave arbitraria nello storage condiviso
    (S3/locale). A differenza di save_task_to_shared_storage (che vive in
    BaseWorker e deriva la chiave da source_info/base_seed/num_trees), qui la
    chiave è passata per intero dal chiamante: serve per casi come i chunk di
    alberi in fase di INFERENZA, dove è l'Orchestratore (non il worker) a
    produrre e caricare i byte, quindi la chiave non ha un 'source_info' da
    cui derivare un job_id.
    """
    if environment == "local":
        local_path = os.path.join("./.local_storage", key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, local_path)
        print(f"[{node_name}] [STORAGE] Salvato '{key}' nello storage locale condiviso.")
        return

    bucket = os.environ.get(bucket_env_var, default_bucket)
    size_mb = len(data) / (1024 ** 2)
    print(f"[{node_name}] [STORAGE] Upload '{key}' su S3 ({size_mb:.1f} MB, bucket: {bucket})...")
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"[{node_name}] [STORAGE] Upload di '{key}' completato ({size_mb:.1f} MB).")


def load_bytes_from_shared_storage(key: str, environment: str, node_name: str = "",
                                    bucket_env_var: str = "DATASETS_BUCKET_NAME",
                                    default_bucket: str = "my-cluster-datasets-bucket-759804778194-us-east-1-an") -> bytes:
    """
    Rilegge byte grezzi da una chiave arbitraria nello storage condiviso.
    Controparte di save_bytes_to_shared_storage. Ritorna None se non trovato
    o in caso di errore.
    """
    if environment == "local":
        local_path = os.path.join("./.local_storage", key)
        if os.path.exists(local_path):
            try:
                with open(local_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"[{node_name}] Errore durante la lettura locale di '{key}': {e}")
        return None

    bucket = os.environ.get(bucket_env_var, default_bucket)
    try:
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        print(f"[{node_name}] [STORAGE HIT] Trovato '{key}' su S3.")
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            print(f"[{node_name}] Errore S3 per la chiave '{key}': {e}")
    except Exception as e:
        print(f"[{node_name}] Errore imprevisto nel recupero della chiave '{key}': {e}")

    return None


def save_task_part_to_shared_storage(source_info: str, base_seed: int, num_trees: int,
                                      part_idx: int, part_trees_bytes: bytes,
                                      environment: str, node_name: str = ""):
    """
    Persiste UN SOLO BATCH (parte) di un task, appena pronto. Il chiamante
    (BaseWorker.exposed_train_subset_forest) invoca questa funzione una
    volta per batch, subito dopo averlo addestrato, cosicché gli alberi di
    quel batch possano essere liberati dalla RAM del worker prima di passare
    al batch successivo, invece di restare accumulati fino alla fine
    dell'intero task (che con chunk grandi e max_depth=None è la causa più
    probabile degli OOM kill osservati sui worker Fargate).
    """
    key = get_task_part_key(source_info, base_seed, num_trees, part_idx)
    save_bytes_to_shared_storage(key, part_trees_bytes, environment, node_name)


def save_task_manifest(source_info: str, base_seed: int, num_trees: int,
                        parts_num_trees: list, environment: str, node_name: str = ""):
    """
    Scrive il manifest DOPO che tutte le parti sono state persistite. Il
    manifest contiene solo interi (numero di alberi per parte), quindi il
    suo costo di memoria/serializzazione è trascurabile rispetto agli alberi
    stessi. La sua presenza è il segnale che l'orchestratore usa per capire
    che il task è completo e ricomponibile (vedi
    load_task_parts_from_shared_storage).
    """
    manifest = {
        "base_seed": base_seed,
        "num_trees": num_trees,
        "num_parts": len(parts_num_trees),
        "parts_num_trees": parts_num_trees,  # alberi per parte, nello stesso ordine dei part_idx
    }
    key = get_task_manifest_key(source_info, base_seed, num_trees)
    save_bytes_to_shared_storage(
        key, json.dumps(manifest).encode("utf-8"), environment, node_name
    )


def load_task_parts_as_tree_list(source_info: str, base_seed: int, num_trees: int,
                                  environment: str, node_name: str = ""):
    """
    Come load_task_parts_from_shared_storage, ma ritorna DIRETTAMENTE la lista
    di alberi già deserializzata (List[DecisionTree...]) invece di bytes
    pickled. Usata da load_task_trees_from_shared_storage per evitare un giro
    di (de)serializzazione superfluo lato chiamante.
    Ritorna None se il manifest non esiste o se manca anche una sola parte.
    """
    manifest_key = get_task_manifest_key(source_info, base_seed, num_trees)
    manifest_bytes = load_bytes_from_shared_storage(manifest_key, environment, node_name)
    if manifest_bytes is None:
        return None

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:
        print(f"[{node_name}] Manifest '{manifest_key}' illeggibile: {e}")
        return None

    all_trees = []
    for part_idx in range(manifest["num_parts"]):
        part_key = get_task_part_key(source_info, base_seed, num_trees, part_idx)
        part_bytes = load_bytes_from_shared_storage(part_key, environment, node_name)
        if part_bytes is None:
            print(f"[{node_name}] [TASK PARTS] Manifest presente ma parte {part_idx} "
                  f"('{part_key}') mancante: task considerato incompleto.")
            return None
        all_trees.extend(pickle.loads(part_bytes))
        # 'part_bytes' esce di scope alla prossima iterazione: non resta mai
        # più di UNA parte serializzata in RAM contemporaneamente, mentre
        # 'all_trees' (gli oggetti già deserializzati) è l'unica struttura
        # che cresce fino a coprire l'intero task -- a differenza della
        # versione precedente, qui non viene MAI ri-serializzata.

    print(f"[{node_name}] [TASK PARTS HIT] Ricomposto task da {manifest['num_parts']} parti "
          f"({len(all_trees)} alberi totali).")
    return all_trees


def load_task_parts_from_shared_storage(source_info: str, base_seed: int, num_trees: int,
                                         environment: str, node_name: str = "") -> bytes:
    """
    Variante 'bytes' di load_task_parts_as_tree_list, mantenuta per i
    chiamanti che si aspettano ancora un blob pickled (es. lo short-circuit
    di BaseWorker.exposed_train_subset_forest, dove il contenuto non viene
    mai realmente decodificato, solo controllato con 'is not None'). NON
    usare questa funzione sul percorso caldo di ricomposizione
    dell'Orchestratore: il pickle.dumps() qui sotto duplica temporaneamente
    in RAM l'intero task appena ricomposto -- usare
    load_task_trees_from_shared_storage in quel caso.
    """
    all_trees = load_task_parts_as_tree_list(source_info, base_seed, num_trees, environment, node_name)
    if all_trees is None:
        return None
    return pickle.dumps(all_trees)


def load_task_trees_from_shared_storage(source_info: str, base_seed: int, num_trees: int,
                                         environment: str, node_name: str = "") -> list:
    """
    Punto d'ingresso CONSIGLIATO per chi ha bisogno degli alberi come oggetti
    Python (l'Orchestratore, in fase di ricomposizione della foresta): prova
    prima il formato monolitico legacy (un solo blob, un solo pickle.loads),
    poi quello a parti (load_task_parts_as_tree_list, senza round-trip di
    serializzazione). Ritorna una lista di alberi, o None se non trovato in
    nessuno dei due formati.
    """
    local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key = get_task_storage_paths(
        source_info, base_seed, num_trees
    )

    if environment == "local":
        if os.path.exists(local_bin_path):
            try:
                with open(local_bin_path, "rb") as f:
                    return pickle.loads(f.read())
            except Exception as e:
                print(f"[{node_name}] Errore durante la lettura del task binario locale: {e}")
    else:
        try:
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            print(f"[{node_name}] [TASK HIT] Trovato task su S3: s3://{s3_bucket}/{s3_key}")
            return pickle.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                print(f"[{node_name}] Errore S3 per il task seed {base_seed}: {e}")
        except Exception as e:
            print(f"[{node_name}] Errore imprevisto nel recupero del task da S3: {e}")

    return load_task_parts_as_tree_list(source_info, base_seed, num_trees, environment, node_name)


def load_task_from_shared_storage(source_info: str, base_seed: int, num_trees: int,
                                   environment: str, node_name: str = "") -> bytes:
    """
    Tenta di recuperare i byte serializzati dell'INTERO TASK dallo storage
    condiviso. Usabile sia dal worker (short-circuit) sia dall'orchestratore
    (rilettura post-ack, invece del ritorno RPC diretto).

    Prova PRIMA il formato monolitico legacy (un solo blob per l'intero
    task), poi quello a parti (manifest + N batch). Questo ordine garantisce
    retrocompatibilità con eventuali task già persistiti dal codice
    precedente al momento del deploy di questa modifica, senza richiedere
    una migrazione: i task vecchi restano leggibili, i nuovi vengono scritti
    (e riletti) a parti.

    Ritorna None se non trovato in nessuno dei due formati o in caso di errore.
    """
    local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key = get_task_storage_paths(
        source_info, base_seed, num_trees
    )

    if environment == "local":
        if os.path.exists(local_bin_path):
            try:
                with open(local_bin_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"[{node_name}] Errore durante la lettura del task binario locale: {e}")
    else:
        try:
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            print(f"[{node_name}] [TASK HIT] Trovato task su S3: s3://{s3_bucket}/{s3_key}")
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                print(f"[{node_name}] Errore S3 per il task seed {base_seed}: {e}")
        except Exception as e:
            print(f"[{node_name}] Errore imprevisto nel recupero del task da S3: {e}")

    # Fallback: formato a parti (nuovo schema di persistenza incrementale).
    return load_task_parts_from_shared_storage(source_info, base_seed, num_trees, environment, node_name)