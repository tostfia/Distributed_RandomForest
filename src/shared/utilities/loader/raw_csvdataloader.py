import os
import glob
import posixpath
import random
from typing import List, Sequence, Union

import fsspec
import pandas as pd

from src.shared.utilities.loader.datasetLoader import DatasetLoader
from src.shared.config import SystemConfig
from src.dataset.dataset_dao import AwsS3DAO, LocalFileSystemDAO


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
    """

    def __init__(
        self,
        data_url: Union[str, Sequence[str]],
        sample_fraction: float = 1.0,
        dataset_seed: int = 123,
        s3_anon: bool = False,
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
        print(f" • Sample fraction:  {self.sample_fraction}")
        print(f" • Dataset seed:     {self.dataset_seed}")
        print(f" • Cache locale:     {'attiva' if self._cache_enabled else 'disattiva'}")

        random.seed(self.dataset_seed)

        for source in sources:
            print(f"   - Lettura e conversione sorgente: {source}")
            df_temp = self._read_single_csv(source=source)
            chunks.append(df_temp)

        if not chunks:
            raise ValueError("Nessun DataFrame caricato.")

        df = pd.concat(chunks, ignore_index=True)

        print("\n[OK] Caricamento CSV grezzo completato.")
        print(f" • Numero totale di righe:   {df.shape[0]}")
        print(f" • Numero totale di colonne: {df.shape[1]}")

        return df

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

    def _read_single_csv(self, source: str) -> pd.DataFrame:
        try:
            # 1. Lettura delegata al DAO corretto in base al tipo di sorgente
            is_s3_source = self._is_s3_path(source)
            dao = self._s3_dao if is_s3_source else self._local_dao
            df_temp = dao.load_dataset(source)

            # 2. Standardizzazione colonne
            df_temp.columns = [c.strip() for c in df_temp.columns]

            # 3. Uniformiamo il nome del target
            if "label" in df_temp.columns:
                df_temp = df_temp.rename(columns={"label": "Label"})

            cols_to_convert = df_temp.columns.difference(["Label"])
            df_temp[cols_to_convert] = df_temp[cols_to_convert].apply(pd.to_numeric, errors="coerce")

            # 4. Cache locale: solo se attiva (ambiente locale) e sorgente S3
            if is_s3_source and self._cache_enabled:
                filename = os.path.basename(source)
                local_cache_path = os.path.join(self.cache_dir, filename)
                if not os.path.exists(local_cache_path):
                    print(f"     [CACHE] Salvo una copia locale di {filename} per i prossimi test...")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    df_temp.to_csv(local_cache_path, index=False)

            # 5. Campionamento in memoria
            if self.sample_fraction < 1.0:
                df_temp = df_temp.sample(frac=self.sample_fraction, random_state=self.dataset_seed)

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