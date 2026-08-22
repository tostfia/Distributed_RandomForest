"""
Script di provisioning STANDALONE per l'ambiente LOCALE del training federato.

È il gemello locale di script_aws/provision_federated_shards.py: da eseguire
UNA VOLTA, PRIMA di avviare master e worker in locale/Docker, per generare su
disco tutto ciò che i worker devono già possedere quando nascono, ovvero gli
shard del dataset reale, uno per ciascun worker:

    ./workers_cache/Worker-Locale-{NN}/train_shard.csv
    ./workers_cache/Worker-Locale-{NN}/test_shard.csv

Perché esiste
-------------
Finora, in locale, lo sharding avveniva "a runtime" dentro
FederatedOrchestrator._ensure_local_bootstrap(), invocato dal primo
_execute_training_step di un job. Questo intreccia la preparazione dei dati con
l'esecuzione del job e, di riflesso, con i test di failover: un job (o un test)
poteva trovarsi a rigenerare gli shard nel bel mezzo dell'esecuzione.

Spostando la generazione qui, il modello locale si allinea a quello AWS: i dati
risiedono già sui nodi quando il sistema parte, e il coordinatore
(FederatedOrchestrator) si limita a VERIFICARE che il provisioning sia stato
fatto — non lo esegue più reattivamente. Vedi _ensure_local_bootstrap (che ora
è un semplice check) e _ensure_aws_bootstrap (già così da prima).

Coerenza con il runtime
-----------------------
I parametri qui sotto (sample_fraction, dataset_seed, target_column, test_size,
random_state, numero di worker, cartella di destinazione, schema di naming
Worker-Locale-NN) DEVONO restare identici a quelli che il runtime si aspetta,
altrimenti i worker non troverebbero i propri shard o li troverebbero diversi
da quelli attesi. Sono gli stessi valori storicamente hard-coded in
_ensure_local_bootstrap.

Uso tipico
----------
    python -m script_local.provision_local_shards --num-workers 7
    python -m script_local.provision_local_shards --num-workers 7 --data-folder ./dataset_cache --force

Per il dataset 'synthetic' non serve provisioning: ogni worker genera
autonomamente il proprio shard sintetico al boot (come già avviene a runtime).
"""
import argparse
import os

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.federated_data_splitter import FederatedDataSplitter

# Parametri di generazione: IDENTICI a quelli usati storicamente da
# FederatedOrchestrator._ensure_local_bootstrap. Non vanno cambiati qui senza
# aggiornare di pari passo il check nel runtime, o i worker riceverebbero shard
# incoerenti con ciò che si aspettano.
SAMPLE_FRACTION = 0.05
DATASET_SEED = 123
TARGET_COLUMN = "Label"
TEST_SIZE = 0.20
RANDOM_STATE = 123

# Default storico: partizionamento IID, invariato rispetto a prima. "dirichlet" e
# "by_day" sono opt-in via CLI/env e servono per simulare eterogeneità non-IID tra
# i worker (vedi FederatedDataSplitter.split_and_shard). alpha è un iperparametro
# dell'ESPERIMENTO (non del modello) e va quindi tracciato insieme al resto della
# configurazione dell'esperimento (es. nel config JSON prodotto da run_baseline.py),
# non hard-codato qui come gli altri parametri sopra.
DEFAULT_PARTITION_STRATEGY = "iid"
DEFAULT_ALPHA = 0.5

BASE_CACHE_DIR = "./workers_cache"

# Cartella predefinita del dataset reale. Stesso nome di variabile d'ambiente
# usato da run_baseline.py, così le due parti del sistema si configurano allo
# stesso modo invece di usare tre nomi diversi ('dataset_path' inesistente su
# SystemConfig, 'DATASET_PATH' qui, nessuno nella baseline).
DEFAULT_DATA_FOLDER = "./dataset_cache"


def _worker_shard_dir(base_cache_dir: str, index_one_based: int) -> str:
    """
    Replica esattamente lo schema di naming usato da FederatedDataSplitter nel
    ramo 'local' (padding a 2 cifre per i primi 9 worker, senza padding dal
    decimo in poi): Worker-Locale-01 ... Worker-Locale-09, Worker-Locale-10, ...
    """
    i = index_one_based - 1
    worker_id = f"Worker-Locale-0{i + 1}" if i < 9 else f"Worker-Locale-{i + 1}"
    return os.path.join(base_cache_dir, worker_id)


def _shards_already_present(num_workers: int, base_cache_dir: str = BASE_CACHE_DIR) -> bool:
    """
    Verifica la presenza degli shard con la STESSA logica di _ensure_local_bootstrap:
    accetta sia la forma con padding (Worker-Locale-01) sia quella senza
    (Worker-Locale-1), per non rigenerare shard validi già prodotti in passato.
    """
    for i in range(1, num_workers + 1):
        dir_padded = os.path.join(base_cache_dir, f"Worker-Locale-{i:02d}")
        dir_unpadded = os.path.join(base_cache_dir, f"Worker-Locale-{i}")

        train_p = os.path.join(dir_padded, "train_shard.csv")
        test_p = os.path.join(dir_padded, "test_shard.csv")
        train_up = os.path.join(dir_unpadded, "train_shard.csv")
        test_up = os.path.join(dir_unpadded, "test_shard.csv")

        if not ((os.path.exists(train_p) and os.path.exists(test_p)) or
                (os.path.exists(train_up) and os.path.exists(test_up))):
            return False
    return True


def _resolve_data_folder(data_folder: str) -> str:
    """
    Cartella sorgente del dataset reale.

    La versione precedente era:

        if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
            return "./dataset_cache" if os.path.exists("./dataset_cache") else "./data"

    cioè, in caso di percorso mancante o inesistente, ripiegava in silenzio su
    './data'. Quel fallback è stato rimosso, qui come in run_baseline.py, per
    due motivi:

      1) './data' non contiene il dataset reale ma è la cartella dove, per
         ragioni storiche, potevano trovarsi CSV estranei: RawCSVDataLoader li
         avrebbe raccolti e trattati come traffico CICIDS, generando shard
         sbagliati senza alcun errore;

      2) un fallback silenzioso su una cartella arbitraria fa fallire il
         provisioning in modo invisibile — i worker ricevono comunque degli
         shard, solo costruiti dai dati sbagliati.

    Ora un percorso assente o inesistente è un errore esplicito.
    """
    resolved = data_folder or os.environ.get("DATASET_LOCAL_PATH", DEFAULT_DATA_FOLDER)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Cartella del dataset reale non trovata: '{resolved}'. "
            f"Posiziona lì i CSV del CICIDS, oppure indica un percorso diverso con "
            f"--data-folder o con la variabile d'ambiente DATASET_LOCAL_PATH. "
            f"(Nessun fallback automatico: generare gli shard da una cartella non "
            f"prevista produrrebbe worker addestrati su dati sbagliati.)"
        )
    return resolved


def provision(num_workers: int, data_folder: str, dataset_type: str = "real", force: bool = False,
              partition_strategy: str = DEFAULT_PARTITION_STRATEGY, alpha: float = DEFAULT_ALPHA,
              day_column: str = None) -> None:
    print("=====================================================")
    print("   PROVISIONING FEDERATO IN LOCALE (offline, one-shot)")
    print("=====================================================")
    print(f" • Worker target:  {num_workers}")
    print(f" • Dataset type:   {dataset_type}")
    print(f" • Strategia part.:{partition_strategy}" + (f" (alpha={alpha})" if partition_strategy == "dirichlet" else ""))
    print(f" • Destinazione:   {os.path.abspath(BASE_CACHE_DIR)}")

    if dataset_type == "synthetic":
        print("=====================================================\n")
        print("[PROVISIONING] Dataset SINTETICO: nessun provisioning necessario. "
              "Ogni worker genera autonomamente il proprio shard sintetico al boot.")
        return

    resolved_folder = _resolve_data_folder(data_folder)
    print(f" • Sorgente dati:  {resolved_folder}")
    print("=====================================================\n")

    if not force and _shards_already_present(num_workers):
        print("[PROVISIONING] Shard già presenti su disco per tutti i worker richiesti. "
              "Salto la rigenerazione (usa --force per sovrascrivere).")
        return

    print("[PROVISIONING] Generazione degli shard in corso...")
    data_loader = RawCSVDataLoader(
        data_url=resolved_folder,
        sample_fraction=SAMPLE_FRACTION,
        dataset_seed=DATASET_SEED,
    )
    splitter = FederatedDataSplitter(
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    splitter.split_and_shard(
        data_loader,
        num_workers=num_workers,
        environment="local",
        partition_strategy=partition_strategy,
        alpha=alpha,
        day_column=day_column,
    )
    print(f"\n[PROVISIONING OK] Shard reali distribuiti nelle cartelle locali dei worker "
          f"sotto '{BASE_CACHE_DIR}'. Il cluster locale è pronto per l'avvio.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provisioning offline degli shard federati su filesystem locale "
                    "(gemello locale di provision_federated_shards.py)."
    )
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", 3)))
    # Allineato a run_baseline.py: stessa variabile d'ambiente DATASET_LOCAL_PATH.
    # Prima il default leggeva 'DATASET_PATH', un terzo nome diverso sia da quello
    # usato dalla baseline sia dall'attributo (inesistente) cercato su SystemConfig.
    parser.add_argument("--data-folder", type=str,
                        default=os.environ.get("DATASET_LOCAL_PATH", DEFAULT_DATA_FOLDER))
    parser.add_argument("--dataset-type", type=str, default=os.environ.get("DATASET_TYPE", "real"),
                        choices=["real", "synthetic"])
    parser.add_argument("--force", action="store_true",
                        help="Rigenera e sovrascrive gli shard anche se già presenti su disco.")
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
        dataset_type=args.dataset_type,
        force=args.force,
        partition_strategy=args.partition_strategy,
        alpha=args.alpha,
        day_column=args.day_column,
    )


if __name__ == "__main__":
    main()