import os
import glob
import random
from typing import List, Sequence, Union

import fsspec
import pandas as pd

from src.shared.utilities.loader.datasetLoader import DatasetLoader

class RawCSVDataLoader(DatasetLoader):
    """
    Loader CSV grezzo per sorgenti locali o S3.
    Supporta una CACHE locale automatica per evitare download ripetuti da S3.
    """

    def __init__(
        self,
        data_url: Union[str, Sequence[str]],
        sample_fraction: float = 1.0,
        dataset_seed: int = 123,
        s3_anon: bool = True,
    ):
        self.data_url = data_url
        self.sample_fraction = float(sample_fraction)
        self.dataset_seed = dataset_seed
        self.s3_anon = s3_anon
        
        # Definiamo la cartella locale dove salvare i file S3 per i test successivi
        self.cache_dir = "./dataset_cache"

        self._validate_parameters()

    def load(self) -> pd.DataFrame:
        sources = self._discover_sources()
        chunks = []

        print("Caricamento dataset CSV grezzo...")
        print(f" • Sorgenti trovate: {len(sources)}")
        print(f" • Sample fraction:  {self.sample_fraction}")
        print(f" • Dataset seed:     {self.dataset_seed}")

        random.seed(self.dataset_seed)

        if self.sample_fraction < 1.0:
            skip_logic = lambda i: i > 0 and random.random() > self.sample_fraction
        else:
            skip_logic = None

        for source in sources:
            print(f"   - Lettura sorgente: {source}")

            # Leggiamo il file (da S3 o da cache locale a seconda di cosa ha deciso _discover_sources)
            df_temp = self._read_single_csv(
                source=source,
                skip_logic=skip_logic
            )
            chunks.append(df_temp)

            # S3 CACHING LOGIC: Se stavamo leggendo da S3, salviamo il file INTERO (senza skip_logic)
            # localmente nella cache per la prossima volta, così la baseline e i futuri test saranno fulminei.
            if self._is_s3_path(source):
                filename = os.path.basename(source)
                local_cache_path = os.path.join(self.cache_dir, filename)
                if not os.path.exists(local_cache_path):
                    print(f"     [CACHE] Salvo una copia locale di {filename} per i prossimi test...")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    # Riscarichiamo il file intero e lo salviamo su disco
                    storage_options = {"anon": self.s3_anon}
                    df_full = pd.read_csv(source, low_memory=False, storage_options=storage_options)
                    df_full.to_csv(local_cache_path, index=False)

        if not chunks:
            raise ValueError("Nessun DataFrame caricato.")

        df = pd.concat(chunks, ignore_index=True)

        print("\n[OK] Caricamento CSV grezzo completato.")
        print(f" • Numero totale di righe:   {df.shape[0]}")
        print(f" • Numero totale di colonne: {df.shape[1]}")

        return df

    def _discover_sources(self) -> List[str]:
        """
        Determina la lista di sorgenti. Se rileva una richiesta S3 ma i file
        sono già presenti nella cache locale, devia la lettura sul disco locale.
        """
        if isinstance(self.data_url, (list, tuple)):
            sources = list(self.data_url)
            
        elif isinstance(self.data_url, str) and self._is_s3_path(self.data_url):
            
            # CONTROLLO CACHE: se la cartella esiste e contiene già i 10 file .csv, usiamo quelli!
            if os.path.exists(self.cache_dir) and len(glob.glob(os.path.join(self.cache_dir, "*.csv"))) >= 10:
                print(f"\n[CACHE HIT] Rilevati file locali in '{self.cache_dir}'. Evito il download da S3.")
                sources = glob.glob(os.path.join(self.cache_dir, "*.csv"))
            else:
                # Se la cache è vuota, andiamo su internet
                if self.data_url.endswith("/"):
                    print(f"[S3 DISCOVERY] Cache vuota o incompleta. Scansione directory Cloud: {self.data_url}")
                    try:
                        fs = fsspec.filesystem("s3", anon=self.s3_anon)
                        raw_files = fs.glob(os.path.join(self.data_url.replace("s3://", ""), "*.csv"))
                        sources = [f"s3://{f}" for f in raw_files]
                    except Exception as e:
                        raise IOError(f"Impossibile listare la cartella S3 {self.data_url}: {e}")
                else:
                    sources = [self.data_url]
                
        elif isinstance(self.data_url, str) and os.path.isdir(self.data_url):
            sources = glob.glob(os.path.join(self.data_url, "*.csv"))
            
        elif isinstance(self.data_url, str):
            sources = [self.data_url]
            
        else:
            raise TypeError("data_url deve essere una stringa o una sequenza di stringhe.")

        sources = sorted(sources)

        if not sources:
            raise FileNotFoundError(f"Nessuna sorgente CSV trovata in: {self.data_url}")

        for source in sources:
            if not self._is_s3_path(source) and not os.path.isfile(source):
                raise FileNotFoundError(f"File locale non trovato: {source}")

        return sources

    def _read_single_csv(self, source: str, skip_logic: callable) -> pd.DataFrame:
        storage_options = None
        if self._is_s3_path(source):
            storage_options = {"anon": self.s3_anon}

        try:
            return pd.read_csv(
                source,
                skiprows=skip_logic,
                low_memory=False,
                storage_options=storage_options,
            )
        except Exception as exc:
            raise IOError(f"Errore nella lettura della sorgente '{source}': {exc}")

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        return isinstance(path, str) and path.startswith("s3://")

    def _validate_parameters(self) -> None:
        if not isinstance(self.dataset_seed, int):
            raise TypeError("dataset_seed deve essere un intero.")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction deve essere nel range (0.0, 1.0].")