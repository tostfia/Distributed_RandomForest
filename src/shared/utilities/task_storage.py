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
import boto3
from botocore.exceptions import ClientError


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
    filename = os.path.basename(source_info)  # es: shared_train_12345.csv
    job_id = filename.replace("shared_train_", "").replace(".csv", "")

    local_dir = os.path.join("./.local_storage", "trained_tasks")
    base_name = f"task_{job_id}_seed_{base_seed}_trees_{num_trees}"
    local_meta_path = os.path.join(local_dir, base_name + ".meta.json")
    local_bin_path = os.path.join(local_dir, base_name + ".bin")

    s3_bucket = os.environ.get(bucket_env_var, default_bucket)
    s3_key = f"tasks/{job_id}/task_seed_{base_seed}_trees_{num_trees}.pkl"

    return local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key


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


def load_task_from_shared_storage(source_info: str, base_seed: int, num_trees: int,
                                   environment: str, node_name: str = "") -> bytes:
    """
    Tenta di recuperare i byte serializzati dell'INTERO TASK dallo storage
    condiviso. Usabile sia dal worker (short-circuit) sia dall'orchestratore
    (rilettura post-ack, invece del ritorno RPC diretto).
    Ritorna None se non trovato o in caso di errore.
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
        return None

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

    return None