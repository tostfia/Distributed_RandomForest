import os
import re
import glob
import json
import posixpath
import random
from typing import List, Optional, Sequence, Union

import fsspec
import pandas as pd

from src.shared.utilities.loader.datasetLoader import DatasetLoader
from src.shared.config import SystemConfig
from src.dataset.dataset_dao import AwsS3DAO, LocalFileSystemDAO


# Colonna usata per taggare ogni riga con il giorno di cattura di
# provenienza (es. "Thuesday-20-02-2018"), estratto dal nome del file
# sorgente. Non è una feature del traffico: va sempre esclusa dal training
# (stesso trattamento delle colonne di metadata in CICIDSPreprocessor) e
# serve solo per (a) il campionamento ribilanciato per giorno e (b) lo
# split day-aware diagnostico (vedi dayaware_holdout.py). Prefissata con
# underscore per distinguerla a colpo d'occhio dalle feature vere.
SOURCE_DAY_COLUMN = "_capture_day"

# Pattern dei nomi file CIC-IDS2018, es.
# "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv" ->  "Friday-02-03-2018"
_DAY_PATTERN = re.compile(r"^([A-Za-z]+-\d{2}-\d{2}-\d{4})")


def _extract_capture_day(source: str) -> str:
    """
    Estrae l'etichetta del giorno di cattura dal nome del file sorgente.
    Ritorna il nome file senza estensione se il pattern non combacia
    (es. sorgenti con naming non standard), così il tagging non solleva mai
    eccezioni — nel peggiore dei casi il "giorno" ricostruito è meno preciso,
    ma il campo resta comunque popolato e utilizzabile per raggruppare le
    righe per file sorgente.
    """
    basename = os.path.basename(source)
    match = _DAY_PATTERN.match(basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


class RawCSVDataLoader(DatasetLoader):
    """
    Loader CSV grezzo per sorgenti locali o S3.

    Il DAO usato per LEGGERE ogni singola sorgente viene scelto in base al
    TIPO di path (s3:// vs locale), non in base all'ambiente di deploy:
    questo loader può girare sia in locale che su AWS e può ricevere
    sorgenti miste (es. test in locale che puntano a un bucket S3).

    La cache locale dei CSV scaricati da S3 resta invece legata
    all'ambiente di SystemConfig: ha senso solo quando si lavora in
    locale (filesystem persistente tra un'esecuzione e l'altra),
    mentre su AWS/Fargate lo storage dei container è effimero e
    scrivere cache su disco lì non porta alcun beneficio, anzi rischia
    di riempire lo storage del task.

    CAMPIONAMENTO RIBILANCIATO PER GIORNO (target_rows_per_day) --
    CIC-IDS2018 è composto da 10 CSV (uno per giorno di cattura) di volume
    molto disomogeneo: un solo giorno (Thuesday-20-02-2018, traffico DDoS)
    può rappresentare da solo quasi la metà dell'intero dataset. Campionando
    ogni file alla STESSA sample_fraction (comportamento di default, invariato
    se target_rows_per_day non è specificato), quello squilibrio si propaga
    identico nel campione: il train set risultante è dominato da un singolo
    contesto di cattura, il che tende a ridurre la varietà tra gli alberi
    della foresta (più alberi "vedono" pattern simili -> correlazione ρ più
    alta -> convergenza OOB più rapida su un problema che appare più facile
    di quanto sarebbe con un mix di giorni più equilibrato — vedi Breiman
    2001, errore atteso della foresta ≈ ρ·σ²).

    Se target_rows_per_day è specificato, il loader:
      1. conta le righe di ciascun file sorgente (DatasetDAO.count_rows,
         SENZA scaricare/parsare l'intero contenuto — per S3 via S3 Select
         lato server, per file locali via conteggio di righe grezzo);
      2. calcola una sample_fraction PER-FILE tale da ottenere circa
         target_rows_per_day righe da ciascuna sorgente (fraction=1.0, cioè
         tutto il file, se il file ha meno righe del target);
      3. campiona ogni file con la propria fraction calcolata, invece della
         stessa fraction globale per tutti.
    I conteggi vengono cachati su disco (row_counts_cache.json) per non
    ripetere il conteggio ad ogni run — rilevante soprattutto su S3, dove
    ogni conteggio è comunque una chiamata di rete, seppur economica.

    TAGGING DEL GIORNO DI CATTURA (tag_source_day) -- se True, aggiunge la
    colonna SOURCE_DAY_COLUMN ad ogni riga con il giorno di provenienza.
    Necessario per lo split day-aware diagnostico (dayaware_holdout.py) a
    valle del caricamento. La colonna NON è una feature: va rimossa prima
    di qualunque fit/training (fatto esplicitamente in dayaware_holdout.py
    e, per sicurezza, va comunque intercettata da CICIDSPreprocessor come le
    altre colonne di metadata se dovesse sopravvivere fino a quel punto —
    vedi nota nel modulo dayaware_holdout).
    """

    ROW_COUNT_CACHE_PATH = os.path.join("./.local_storage", "row_counts_cache.json")

    def __init__(
        self,
        data_url: Union[str, Sequence[str]],
        sample_fraction: float = 1.0,
        dataset_seed: int = 123,
        s3_anon: bool = False,
        target_rows_per_day: Optional[int] = None,
        tag_source_day: bool = False,
    ):
        if isinstance(data_url, str):
            self.data_url = data_url.strip().strip("'\"")
        elif isinstance(data_url, (list, tuple)):
            self.data_url = [str(u).strip().strip("'\"") for u in data_url]
        else:
            self.data_url = data_url

        self.sample_fraction = float(sample_fraction)
        self.dataset_seed = dataset_seed
        self.s3_anon = s3_anon
        self.target_rows_per_day = target_rows_per_day
        self.tag_source_day = tag_source_day

        # DAO dedicati per tipo di sorgente (non per ambiente di deploy)
        self._local_dao = LocalFileSystemDAO()
        self._s3_dao = AwsS3DAO()

        # La cache su disco dei file S3 è utile solo in locale, per non
        # riscaricare ad ogni run di test. Su AWS resta sempre disattiva.
        env = SystemConfig().env.strip().lower()
        self._cache_enabled = env == "local"
        self.cache_dir = "./dataset_cache"

        self._validate_parameters()

    def load(self) -> pd.DataFrame:
        sources = self._discover_sources()
        chunks = []

        print("Caricamento dataset CSV grezzo...")
        print(f" • Sorgenti trovate: {len(sources)}")
        if self.target_rows_per_day is not None:
            print(f" • Campionamento: RIBILANCIATO per giorno (target ~{self.target_rows_per_day} "
                  f"righe/file, prima del conteggio effettivo per sorgente)")
        else:
            print(f" • Sample fraction:  {self.sample_fraction} (uniforme su tutte le sorgenti)")
        print(f" • Dataset seed:     {self.dataset_seed}")
        print(f" • Cache locale:     {'attiva' if self._cache_enabled else 'disattiva'}")
        if self.tag_source_day:
            print(f" • Tagging giorno di cattura: attivo (colonna '{SOURCE_DAY_COLUMN}')")

        random.seed(self.dataset_seed)

        row_counts = None
        if self.target_rows_per_day is not None:
            row_counts = self._get_row_counts(sources)

        for source in sources:
            per_source_fraction = self.sample_fraction
            if self.target_rows_per_day is not None:
                total_rows = row_counts.get(source, 0)
                per_source_fraction = (
                    min(1.0, self.target_rows_per_day / total_rows) if total_rows > 0 else 1.0
                )
                print(f"   - Lettura e conversione sorgente: {source} "
                      f"({total_rows} righe totali -> fraction={per_source_fraction:.4f})")
            else:
                print(f"   - Lettura e conversione sorgente: {source}")

            df_temp = self._read_single_csv(source=source, sample_fraction=per_source_fraction)
            chunks.append(df_temp)

        if not chunks:
            raise ValueError("Nessun DataFrame caricato.")

        df = pd.concat(chunks, ignore_index=True)

        print("\n[OK] Caricamento CSV grezzo completato.")
        print(f" • Numero totale di righe:   {df.shape[0]}")
        print(f" • Numero totale di colonne: {df.shape[1]}")

        if self.target_rows_per_day is not None and self.tag_source_day:
            print("\n • Distribuzione righe per giorno di cattura (dopo il campionamento):")
            for day, count in df[SOURCE_DAY_COLUMN].value_counts().sort_index().items():
                print(f"     {day:<25} {count:>8} righe")

        return df

    def _get_row_counts(self, sources: List[str]) -> dict:
        """
        Ritorna {source: n_righe} per tutte le sorgenti, usando una cache su
        disco (ROW_COUNT_CACHE_PATH) per evitare di ricontare ad ogni run —
        rilevante soprattutto su S3, dove ogni conteggio (anche se economico
        via S3 Select) è comunque una chiamata di rete.

        La cache è invalidata SOLO per le singole sorgenti mancanti o non più
        presenti: non viene mai cancellata per intero, così l'aggiunta di un
        nuovo CSV alla cartella non costringe a riconteggiare tutti gli
        altri.
        """
        cache = {}
        if os.path.exists(self.ROW_COUNT_CACHE_PATH):
            try:
                with open(self.ROW_COUNT_CACHE_PATH, "r") as f:
                    cache = json.load(f)
            except Exception as e:
                print(f"   [ATTENZIONE] Cache dei row count illeggibile ({e}), la ricostruisco da zero.")
                cache = {}

        missing = [s for s in sources if s not in cache]
        if missing:
            print(f"   [ROW COUNT] {len(missing)} sorgente/i senza conteggio in cache, "
                  f"le conto ora (S3 Select lato server per le sorgenti S3)...")
            for source in missing:
                is_s3_source = self._is_s3_path(source)
                dao = self._s3_dao if is_s3_source else self._local_dao
                cache[source] = dao.count_rows(source)
                print(f"     - {source}: {cache[source]} righe")

            os.makedirs(os.path.dirname(self.ROW_COUNT_CACHE_PATH), exist_ok=True)
            with open(self.ROW_COUNT_CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2)
        else:
            print(f"   [ROW COUNT] Tutte le {len(sources)} sorgenti già in cache "
                  f"('{self.ROW_COUNT_CACHE_PATH}').")

        return {s: cache[s] for s in sources}

    def _discover_sources(self) -> List[str]:
        """
        Determina la lista di sorgenti ordinata alfabeticamente per replicabilità del seed.
        """
        if isinstance(self.data_url, (list, tuple)):
            sources = sorted(list(self.data_url))

        elif isinstance(self.data_url, str) and self._is_s3_path(self.data_url):
            # Cache hit: solo se siamo in locale e abbiamo già almeno 10 CSV cachati
            if (
                self._cache_enabled
                and os.path.exists(self.cache_dir)
                and len(glob.glob(os.path.join(self.cache_dir, "*.csv"))) >= 10
            ):
                print(f"\n[CACHE HIT] Rilevati file locali in '{self.cache_dir}'. Evito il download da S3.")
                sources = sorted(glob.glob(os.path.join(self.cache_dir, "*.csv")))
            else:
                if self.data_url.endswith("/"):
                    print(f"[S3 DISCOVERY] Scansione directory Cloud: {self.data_url}")
                    try:
                        fs = fsspec.filesystem("s3", anon=self.s3_anon)
                        clean_url = self.data_url.replace("s3://", "")
                        search_pattern = posixpath.join(clean_url, "*.csv")
                        raw_files = fs.glob(search_pattern)

                        sources = []
                        for f in raw_files:
                            if f.startswith("s3://"):
                                sources.append(f)
                            else:
                                sources.append(f"s3://{f}")
                        sources = sorted(sources)
                    except Exception as e:
                        raise IOError(f"Impossibile listare la cartella S3 {self.data_url}: {e}")
                else:
                    sources = [self.data_url]

        elif isinstance(self.data_url, str) and os.path.isdir(self.data_url):
            sources = sorted(glob.glob(os.path.join(self.data_url, "*.csv")))

        elif isinstance(self.data_url, str):
            sources = [self.data_url]

        else:
            raise TypeError("data_url deve essere una stringa o una sequenza di stringhe.")

        if not sources:
            raise FileNotFoundError(f"Nessuna sorgente CSV trovata in: {self.data_url}")

        for source in sources:
            if not self._is_s3_path(source) and not os.path.isfile(source):
                raise FileNotFoundError(f"File locale non trovato: {source}")

        return sources

    def _read_single_csv(self, source: str, sample_fraction: Optional[float] = None) -> pd.DataFrame:
        """
        sample_fraction: se fornito, sovrascrive self.sample_fraction per
        QUESTA sorgente (usato dal campionamento ribilanciato per giorno).
        Se None, usa self.sample_fraction come prima (comportamento invariato).
        """
        effective_fraction = self.sample_fraction if sample_fraction is None else sample_fraction
        try:
            # 1. Lettura delegata al DAO corretto in base al tipo di sorgente.
            # Il campionamento (se richiesto) avviene DENTRO il DAO, in streaming
            # chunk-per-chunk, così non si carica mai l'intero file in RAM prima
            # di scartarne il 99%.
            is_s3_source = self._is_s3_path(source)
            dao = self._s3_dao if is_s3_source else self._local_dao
            df_temp = dao.load_dataset(
                source,
                sample_fraction=effective_fraction,
                dataset_seed=self.dataset_seed,
            )

            # 2. Standardizzazione colonne
            df_temp.columns = [c.strip() for c in df_temp.columns]

            # 3. Uniformiamo il nome del target
            if "label" in df_temp.columns:
                df_temp = df_temp.rename(columns={"label": "Label"})

            # 3b. Rimozione di eventuali righe di header duplicato annidate in
            # mezzo al file (tipico dei CSV CIC-IDS2018.
            if "Label" in df_temp.columns:
                header_dupe_mask = df_temp["Label"].astype(str).str.strip() == "Label"
                n_dupes = int(header_dupe_mask.sum())
                if n_dupes:
                    print(
                        f"     [PULIZIA] Rimosse {n_dupes} riga/e di header duplicato "
                        f"(Label=='Label') da '{source}'."
                    )
                    df_temp = df_temp[~header_dupe_mask].reset_index(drop=True)

            # 3c. Tagging del giorno di cattura (opzionale, vedi docstring
            # della classe). Fatto DOPO la pulizia degli header duplicati,
            # così un'eventuale riga di header duplicato non riceve comunque
            # un tag di giorno prima di essere scartata al passo precedente.
            if self.tag_source_day:
                df_temp[SOURCE_DAY_COLUMN] = _extract_capture_day(source)

            # 4. NESSUNA conversione numerica qui. PRIMA veniva fatta anche a
            # questo livello (pd.to_numeric su tutte le colonne tranne
            # "Label"/SOURCE_DAY_COLUMN) -- doppione con l'identica
            # conversione già eseguita a valle da
            # CICIDSPreprocessor._convert_feature_columns_to_numeric, con in
            # più una lista di esclusione diversa (rischio di disallineamento
            # silenzioso tra le due). Rimossa: la tipizzazione numerica resta
            # un'unica responsabilità del preprocessor (chiamato sempre
            # dopo, sia in run_baseline.py sia in dayaware_holdout.py),
            # invece che duplicata qui. Il DataFrame restituito da questo
            # loader può quindi contenere colonne ancora di tipo object/
            # stringa grezza: chi lo consuma direttamente (senza passare da
            # CICIDSPreprocessor) deve convertire esplicitamente prima del
            # training.

            # 5. Cache locale: solo se attiva (ambiente locale) e sorgente S3
            if is_s3_source and self._cache_enabled:
                filename = os.path.basename(source)
                local_cache_path = os.path.join(self.cache_dir, filename)
                if not os.path.exists(local_cache_path):
                    print(f"     [CACHE] Salvo una copia locale di {filename} per i prossimi test...")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    df_temp.to_csv(local_cache_path, index=False)

            return df_temp

        except Exception as exc:
            raise IOError(f"Errore nella lettura/conversione della sorgente '{source}': {exc}")

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        if not isinstance(path, str):
            return False
        return path.strip().startswith("s3://")

    def _validate_parameters(self) -> None:
        if not isinstance(self.dataset_seed, int):
            raise TypeError("dataset_seed deve essere un intero.")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction deve essere nel range (0.0, 1.0].")
        if self.target_rows_per_day is not None and self.target_rows_per_day <= 0:
            raise ValueError("target_rows_per_day deve essere maggiore di 0, se specificato.")