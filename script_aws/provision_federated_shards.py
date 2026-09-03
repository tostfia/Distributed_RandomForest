"""
Script di provisioning STANDALONE per l'ambiente AWS del training federato.

Da eseguire UNA VOLTA, PRIMA di avviare master e worker sul cluster (EC2/Fargate),
per "seminare" su S3 tutto ciò che i worker devono già possedere quando nascono:

  1. gli shard del dataset reale, uno per ciascun worker
     (s3://<bucket>/federated_shards/worker_{i}/{train,test}_shard.csv);
  2. i manifesti di feature selection già generati dalla baseline offline
     (outputs_baseline/config_real.json e/o config_synthetic.json, prodotti da
     src/baseline/run_baseline.py), copiati così come sono sotto
     s3://<bucket>/federated_config/.

Nessuna di queste due cose viene più generata "a runtime" da un job di training:
il coordinatore (FederatedOrchestrator) si limita a VERIFICARE che il
provisioning sia stato fatto, e ogni FederatedWorker, al boot, scarica UNA
VOLTA lo shard di propria competenza e i manifesti disponibili — prima ancora
di registrarsi come disponibile nel ServiceRegistry.

Uso tipico:
    python -m script_aws.provision_federated_shards --num-workers 3 --data-folder ./dataset_cache
    
NOTA: questo script assume che tu abbia già eseguito (se ti interessa quel
dataset_type) src/baseline/run_baseline.py in locale, così che
outputs_baseline/config_real.json e/o config_synthetic.json esistano già sul
tuo filesystem prima del provisioning. Non rilancia la baseline da solo.
"""
import argparse
import os
import boto3
from botocore.exceptions import ClientError

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.federated_data_splitter import FederatedDataSplitter

DEFAULT_BUCKET = os.environ.get(
    "DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an"
)
OUTPUTS_BASELINE_DIR = "outputs_baseline"
# Stesso valore di run_baseline.py/centralized.py/provision_local_shards.py:
# campionamento ribilanciato per giorno invece di sample_fraction=0.05
# uniforme -- vedi provision_local_shards.py per la motivazione completa
# (gemello locale di questo script, deve restare allineato).
TARGET_ROWS_PER_DAY = 100_000

# Default storico: partizionamento IID, invariato rispetto a prima. "dirichlet" e
# "by_day" sono opt-in via CLI/env, gemelle di quelle esposte da
# script_local/provision_local_shards.py (vedi lì per il significato di alpha).
# Se li usi, ricordati di tenerli coerenti con quanto eventualmente registrato
# nel manifesto (config_real.json/config_synthetic.json prodotto da
# src/baseline/run_baseline.py), così com'è già richiesto per gli altri
# parametri di generazione (sample_fraction, seed, ecc.).
DEFAULT_PARTITION_STRATEGY = "iid"
DEFAULT_ALPHA = 0.5

def _shards_already_present(s3_client, bucket: str, num_workers: int) -> bool:
    for i in range(1, num_workers + 1):
        for fname in ("train_shard.csv", "test_shard.csv"):
            key = f"federated_shards/worker_{i}/{fname}"
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
            except ClientError:
                return False
    return True


def _upload_feature_config_manifests(s3_client, bucket: str) -> None:
    """
    Carica su S3, sotto federated_config/, i manifesti config_*.json già
    presenti in locale (generati da src/baseline/run_baseline.py).

    È un'operazione best-effort e indipendente dal dataset_type che l'utente
    finirà per scegliere quando sottomette un job: carichiamo TUTTI i
    manifesti disponibili in locale, così ogni worker li ha già entrambi in
    cache e può servire sia un job 'real' sia uno 'synthetic' senza dover
    ri-fare provisioning.
    """
    for dataset_type in ("real", "synthetic"):
        filename = f"config_{dataset_type}.json"
        local_path = os.path.join(OUTPUTS_BASELINE_DIR, filename)
        if not os.path.exists(local_path):
            print(
                f"[PROVISIONING] '{local_path}' non trovato in locale, salto "
                f"(ok se non ti serve dataset_type='{dataset_type}')."
            )
            continue
        s3_key = f"federated_config/{filename}"
        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"[PROVISIONING] Caricato manifesto '{local_path}' -> s3://{bucket}/{s3_key}")


def provision(num_workers: int, data_folder: str, bucket: str, force: bool = False,
              partition_strategy: str = DEFAULT_PARTITION_STRATEGY, alpha: float = DEFAULT_ALPHA,
              day_column: str = None) -> None:
    s3_client = boto3.client("s3")

    print("=====================================================")
    print("   PROVISIONING FEDERATO SU AWS (offline, one-shot)  ")
    print("=====================================================")
    print(f" • Worker target:  {num_workers}")
    print(f" • Bucket S3:      {bucket}")
    print(f" • Sorgente dati:  {data_folder}")
    print(f" • Strategia part.:{partition_strategy}" + (f" (alpha={alpha})" if partition_strategy == "dirichlet" else ""))
    print("=====================================================\n")

    if not force and _shards_already_present(s3_client, bucket, num_workers):
        print(
            "[PROVISIONING] Shard già presenti su S3 per tutti i worker richiesti. "
            "Salto la rigenerazione (usa --force per sovrascrivere)."
        )
    else:
        print("[PROVISIONING] Generazione e upload degli shard in corso...")
        data_loader = RawCSVDataLoader(
            data_url=data_folder,
            dataset_seed=123,
            target_rows_per_day=TARGET_ROWS_PER_DAY,
        )
        splitter = FederatedDataSplitter(target_column="Label", test_size=0.20, random_state=123)
        splitter.split_and_shard(
            data_loader, num_workers=num_workers, environment="aws", bucket_name=bucket,
            partition_strategy=partition_strategy, alpha=alpha, day_column=day_column,
        )
        print("[PROVISIONING] Shard caricati su S3 con successo.")

    _upload_feature_config_manifests(s3_client, bucket)

    print(
        "\n[PROVISIONING OK] Cluster pronto per l'avvio. Ricorda di impostare, per ciascuna "
        "istanza worker EC2, la variabile d'ambiente WORKER_INDEX (1..N) PRIMA di avviarla "
        "(es. via user-data / launch template): è il binding fisso worker<->shard."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provisioning offline degli shard federati e dei manifesti feature su S3."
    )
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", 3)))
    parser.add_argument("--data-folder", type=str, default=os.environ.get("DATASET_PATH", "./dataset_cache"))
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rigenera e sovrascrive gli shard anche se già presenti su S3.",
    )
    parser.add_argument("--partition-strategy", type=str,
                        default=os.environ.get("PARTITION_STRATEGY", DEFAULT_PARTITION_STRATEGY),
                        choices=["iid", "dirichlet", "by_day"],
                        help="Strategia di partizionamento tra i worker: 'iid' (default, storica), "
                             "'dirichlet' (eterogeneità sintetica controllata da --alpha), "
                             "'by_day' (partizionamento naturale per file/giorno di origine).")
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("ALPHA", DEFAULT_ALPHA)),
                        help="Iperparametro di eterogeneità per partition_strategy='dirichlet'. "
                             "Valori piccoli (es. 0.1) = eterogeneità estrema; valori grandi "
                             "(es. 10+) tendono all'IID.")
    parser.add_argument("--day-column", type=str, default=os.environ.get("DAY_COLUMN"),
                        help="Nome della colonna che identifica il giorno/file di origine, "
                             "richiesta solo con partition_strategy='by_day'.")
    args = parser.parse_args()

    provision(
        num_workers=args.num_workers,
        data_folder=args.data_folder,
        bucket=args.bucket,
        force=args.force,
        partition_strategy=args.partition_strategy,
        alpha=args.alpha,
        day_column=args.day_column,
    )


if __name__ == "__main__":
    main()