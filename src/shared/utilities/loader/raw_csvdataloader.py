import os
import glob
import posixpath
import random
from typing import List, Sequence, Union

import fsspec
import pandas as pd

from src.shared.utilities.loader.datasetLoader import DatasetLoader

class RawCSVDataLoader(DatasetLoader):
    """
    Loader CSV grezzo per sorgenti locali o S3.
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
        
        # Definiamo la cartella locale dove salvare i file S3 per i test successivi
        self.cache_dir = "./dataset_cache"

        self._validate_parameters()

    def load(self) -> pd.DataFrame:
        sources = self._discover_sources()
        chunks = []

        print("Caricamento dataset CSV grezzo (Logica speculare a Colab)...")
        print(f" • Sorgenti trovate: {len(sources)}")
        print(f" • Sample fraction:  {self.sample_fraction}")
        print(f" • Dataset seed:     {self.dataset_seed}")

        random.seed(self.dataset_seed)

        for source in sources:
            print(f"   - Lettura e conversione sorgente: {source}")

            # Lettura UNICA della sorgente (niente più skiprows in fase di parsing: con S3
            # il traffico di rete avviene comunque per l'intero oggetto, quindi non risparmiava
            # banda e obbligava a un secondo download identico per popolare la cache).
            # Il campionamento, se richiesto, viene applicato in memoria DOPO il download,
            # dentro _read_single_csv.
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
            if os.path.exists(self.cache_dir) and len(glob.glob(os.path.join(self.cache_dir, "*.csv"))) >= 10:
                print(f"\n[CACHE HIT] Rilevati file locali in '{self.cache_dir}'. Evito il download da S3.")
                sources = sorted(glob.glob(os.path.join(self.cache_dir, "*.csv"))) 
            else:
                if self.data_url.endswith("/"):
                    print(f"[S3 DISCOVERY] Cache vuota. Scansione directory Cloud: {self.data_url}")
                    try:
                        fs = fsspec.filesystem("s3", anon=self.s3_anon)
                        # Sostituiamo os.path.join con posixpath per garantire gli slash '/' corretti su S3
                        clean_url = self.data_url.replace("s3://", "")
                        search_pattern = posixpath.join(clean_url, "*.csv")
                        raw_files = fs.glob(search_pattern)
                        
                        # Ricostruiamo i path assicurandoci che abbiano il prefisso s3:// corretto
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
            sources = sorted(glob.glob(os.path.join(self.data_url, "*.csv"))) # 🌟 Ordinato
            
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
        storage_options = None
        if self._is_s3_path(source):
            storage_options = {"anon": self.s3_anon}

        try:
            # 1. Download COMPLETO e UNICO della sorgente (senza skiprows)
            df_temp = pd.read_csv(
                source,
                low_memory=False,
                storage_options=storage_options,
            )
            
            # 2. Standardizzazione colonne
            df_temp.columns = [c.strip() for c in df_temp.columns]

            # 3. Uniformiamo il nome del target
            if 'label' in df_temp.columns:
                df_temp = df_temp.rename(columns={'label': 'Label'})

            cols_to_convert  = df_temp.columns.difference(['Label'])
            df_temp[cols_to_convert] = df_temp[cols_to_convert].apply(pd.to_numeric, errors='coerce')

            # 4. S3 CACHING LOGIC: se la sorgente era S3, salviamo su disco il file GIA'
            #    scaricato al passo 1 (invece di riscaricarlo una seconda volta per intero)
            if self._is_s3_path(source):
                filename = os.path.basename(source)
                local_cache_path = os.path.join(self.cache_dir, filename)
                if not os.path.exists(local_cache_path):
                    print(f"     [CACHE] Salvo una copia locale di {filename} per i prossimi test...")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    df_temp.to_csv(local_cache_path, index=False)

            # 5. Campionamento in memoria (sostituisce il vecchio skip_logic basato su skiprows,
            
            if self.sample_fraction < 1.0:
                df_temp = df_temp.sample(frac=self.sample_fraction, random_state=self.dataset_seed)

            return df_temp

        except Exception as exc:
            raise IOError(f"Errore nella lettura/conversione della sorgente '{source}': {exc}")

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        if not isinstance(path, str):
            return False
        # Applichiamo lo strip anche qui per sicurezza
        return path.strip().startswith("s3://")

    def _validate_parameters(self) -> None:
        if not isinstance(self.dataset_seed, int):
            raise TypeError("dataset_seed deve essere un intero.")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction deve essere nel range (0.0, 1.0].")