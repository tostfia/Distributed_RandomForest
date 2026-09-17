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
# split day-aware diagnostico (vedi diagnose_false_negative.py). Prefissata con
# underscore per distinguerla a colpo d'occhio dalle feature vere.
SOURCE_DAY_COLUMN = "_capture_day"

# Pattern dei nomi file CIC-IDS2018, es.
# "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv" ->  "Friday-02-03-2018"
_DAY_PATTERN = re.compile(r"^([A-Za-z]+-\d{2}-\d{2}-\d{4})")

# Giorni di cattura in cui il CIC-IDS2018 include l'attacco "Infiltration"
# (fonte: documentazione ufficiale del dataset, unb.ca/cic/datasets/ids-2018
# -- Infiltration compare SOLO in questi due file, a differenza di
# DDoS/DoS/Bot/Brute Force che ricorrono anche altrove con moltissimi
# esempi). Anche dentro questi due file Infiltration resta una minoranza
# rispetto al Benign catturato lo stesso giorno: un campionamento uniforme
# riga-per-riga (come per gli altri 8 giorni) la dilluirebbe ulteriormente
# PRIMA ancora che split/undersampling possano intervenire -- diagnosticato
# empiricamente in diagnose_false_negatives.py (Infiltration: recall
# 15.56% sul test, 99.4% dei Falsi Negativi totali del modello). Vedi
# RawCSVDataLoader._rebalance_protected_source per il trattamento speciale.
DEFAULT_PROTECTED_MINORITY_DAYS = frozenset({"Wednesday-28-02-2018", "Thursday-01-03-2018"})


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
    Loader del dataset reale (CIC-IDS2018) da CSV grezzi, locali o su S3.

    Oltre alla semplice lettura/concatenazione dei file sorgente, implementa
    due correzioni allo squilibrio naturale del dataset:
      - campionamento RIBILANCIATO per giorno di cattura (target_rows_per_day);
      - trattamento speciale dei giorni con classi di attacco rare.

    Supporta anche il tagging opzionale di ogni riga col proprio giorno di
    cattura (tag_source_day), usato dal partizionamento federato 'by_day' e
    da alcuni script diagnostici, mai una feature di training, va sempre
    rimossa prima del fit.

    ---
    DETTAGLI E SCELTE IMPLEMENTATIVE

    1. DAO e Gestione Cache su AWS:
       Il DAO per la lettura viene scelto in base al TIPO di path (s3:// vs locale), 
       permettendo sorgenti miste. La cache locale dei CSV scaricati da S3, invece, 
       è legata all'ambiente (SystemConfig): si attiva in locale (filesystem persistente), 
       ma resta disattiva su AWS/Fargate dove lo storage dei container è effimero e 
       scrivere su disco rischierebbe solo di saturare lo spazio del task.

    2. Campionamento Ribilanciato (target_rows_per_day):
       I 10 CSV di CIC-IDS2018 hanno volumi disomogenei (un giorno DDoS fa quasi il 50% 
       del totale). Un campionamento uniforme (stessa fraction per tutti) propagherebbe 
       lo squilibrio, facendo dominare il train set da un singolo contesto e riducendo 
       la varietà tra gli alberi (convergenza precoce, correlazione ρ più alta — 
       Breiman 2001). 
       Se `target_rows_per_day` è specificato, il loader conta le righe (usando una cache 
       JSON locale o chiamate S3 Select per non scaricare i file interi) e calcola una 
       `sample_fraction` dinamica PER-FILE.

    3. Giorni "Protetti" (protected_minority_days):
       Eccezione al punto 2. Classi come "Infiltration" compaiono solo in due giorni specifici. 
       Campionare questi giorni in streaming (riga-per-riga) diluirebbe fatalmente una 
       classe già rara prima che lo split/undersampling possano intervenire. 
       Per questi giorni, il loader legge l'intero file, tiene TUTTE le righe non-Benign, 
       e sotto-campiona SOLO il traffico Benign fino a raggiungere il target.
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
        protected_minority_days: Optional[frozenset] = None,
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
        # Overridabile per dataset/versioni future del CICIDS con giorni
        # "rari" diversi; di default i due giorni Infiltration documentati
        # sopra (vedi DEFAULT_PROTECTED_MINORITY_DAYS).
        self.protected_minority_days = (
            protected_minority_days
            if protected_minority_days is not None
            else DEFAULT_PROTECTED_MINORITY_DAYS
        )

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
            is_protected = (
                self.target_rows_per_day is not None
                and _extract_capture_day(source) in self.protected_minority_days
            )

            if self.target_rows_per_day is not None:
                total_rows = row_counts.get(source, 0)

                if is_protected:
                    # Vedi DEFAULT_PROTECTED_MINORITY_DAYS / docstring
                    # classe: nessuna fraction uniforme qui, si legge tutto
                    # e si ribilancia dopo (_rebalance_protected_source).
                    print(f"   - Lettura sorgente PROTETTA (classe minoritaria nota): {source} "
                          f"({total_rows} righe totali -> lettura completa, poi "
                          f"ribilanciamento locale Benign/non-Benign)")
                    df_temp = self._read_single_csv(source=source, sample_fraction=1.0)
                    df_temp = self._rebalance_protected_source(df_temp, source)
                    chunks.append(df_temp)
                    continue

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

    def _rebalance_protected_source(self, df_temp: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Ribilanciamento locale per un file "protetto" (vedi
        DEFAULT_PROTECTED_MINORITY_DAYS / docstring classe): tiene TUTTE le
        righe non-Benign (qualunque sotto-tipo di attacco presente in
        questo file, non solo Infiltration) e sotto-campiona SOLO il
        Benign per restare vicino a target_rows_per_day -- stesso
        principio di undersample_majority_class (undersampling.py), ma
        applicato qui per-singolo-file, PRIMA dello split, così una classe
        già rara non viene ulteriormente diluita da un campionamento in
        streaming cieco al Label.
        """
        if "Label" not in df_temp.columns:
            print(f"     [ATTENZIONE] Colonna 'Label' assente in '{source}': impossibile "
                  f"ribilanciare per classe, file tenuto per intero senza campionamento.")
            return df_temp

        benign_mask = df_temp["Label"].astype(str).str.strip() == "Benign"
        non_benign_df = df_temp[~benign_mask]
        benign_df = df_temp[benign_mask]

        n_non_benign = len(non_benign_df)
        benign_budget = max(0, self.target_rows_per_day - n_non_benign)

        if len(benign_df) > benign_budget:
            benign_df = (
                benign_df.sample(n=benign_budget, random_state=self.dataset_seed)
                if benign_budget > 0
                else benign_df.iloc[0:0]
            )

        result = pd.concat([non_benign_df, benign_df], ignore_index=True)
        # Mescolato (stessa igiene applicata in undersample_majority_class):
        # altrimenti tutte le righe non-Benign finirebbero in blocco
        # all'inizio del file.
        result = result.sample(frac=1.0, random_state=self.dataset_seed).reset_index(drop=True)

        print(f"     [RIBILANCIAMENTO LOCALE] '{source}': {n_non_benign} righe non-Benign "
              f"mantenute INTERE (nessun campionamento), Benign ridotto da "
              f"{(benign_mask).sum()} a {len(benign_df)} righe (budget = "
              f"target_rows_per_day - non_benign = {self.target_rows_per_day} - "
              f"{n_non_benign}). Righe totali risultanti: {len(result)}.")

        return result

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

            # 4. NESSUNA conversione numerica a questo livello.
            # La tipizzazione numerica è responsabilità unica ed esclusiva
            # di CICIDSPreprocessor (chiamato a valle). Il DataFrame restituito
            # conserva quindi i tipi grezzi (object/stringa): chi lo consuma
            # direttamente senza passare dal preprocessor deve gestirne la
            # conversione esplicitamente prima del training.

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