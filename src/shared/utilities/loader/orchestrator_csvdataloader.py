import os
import glob
import random
from typing import List, Sequence, Union

import pandas as pd

from shared.utilities.loader import DatasetLoaderStrategy


class RawCSVDataLoader(DatasetLoaderStrategy):
    """
    Loader CSV grezzo per sorgenti locali o S3.

    Responsabilità:
    - leggere CSV da file system locale o S3;
    - applicare opzionalmente un campionamento deterministico sequenziale (come da baseline);
    - restituire un DataFrame grezzo.
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

        # Validazione dei parametri all'inizio per evitare errori a runtime durante il caricamento.
        self._validate_parameters()


    #Esegue la scansione dei file,applica il campionamento e restituisce un DataFrame concatenato.
    def load(self) -> pd.DataFrame:
        sources = self._discover_sources()
        chunks = []

        print("Caricamento dataset CSV grezzo...")
        print(f" • Sorgenti trovate: {len(sources)}")
        print(f" • Sample fraction:  {self.sample_fraction}")
        print(f" • Dataset seed:     {self.dataset_seed}")

        random.seed(self.dataset_seed)

        #Funzione lambda per il campionamento: salta una riga con probabilità (1 - sample_fraction) dopo la prima riga.
        if self.sample_fraction < 1.0:
            skip_logic = lambda i: i > 0 and random.random() > self.sample_fraction
        else:
            skip_logic = None

        #Lettura sequenziale dei file
        for source in sources:
            print(f"   - Lettura sorgente: {source}")

            # Passiamo la skip_logic direttamente alla funzione di lettura
            df_temp = self._read_single_csv(
                source=source,
                skip_logic=skip_logic
            )

            chunks.append(df_temp)

        if not chunks:
            raise ValueError("Nessun DataFrame caricato.")

        #Unione di tutti i DataFrame caricati in un unico DataFrame finale
        df = pd.concat(chunks, ignore_index=True)

        print("\n[OK] Caricamento CSV grezzo completato.")
        print(f" • Numero totale di righe:   {df.shape[0]}")
        print(f" • Numero totale di colonne: {df.shape[1]}")

        return df

    
    def _discover_sources(self) -> List[str]:
        """
        Determina la lista di sorgenti CSV da leggere.

        Supporta:
        - singolo file locale;
        - directory locale;
        - singolo URL S3;
        - lista di file/URL.

        Nota:
        una directory S3 non viene listata automaticamente.
        Per S3 conviene passare una lista esplicita di file.
        """
        if isinstance(self.data_url, (list, tuple)):
            sources = list(self.data_url)
        elif isinstance(self.data_url, str) and os.path.isdir(self.data_url):
            sources = glob.glob(os.path.join(self.data_url, "*.csv"))
        elif isinstance(self.data_url, str):
            sources = [self.data_url]
        else:
            raise TypeError("data_url deve essere una stringa o una sequenza di stringhe.")

        # Ordine deterministico utile per benchmark e riproducibilità.
        sources = sorted(sources)

        if not sources:
            raise FileNotFoundError(f"Nessuna sorgente CSV trovata in: {self.data_url}")

        for source in sources:
            if not self._is_s3_path(source) and not os.path.isfile(source):
                raise FileNotFoundError(f"File locale non trovato: {source}")

        return sources

    def _read_single_csv(
        self,
        source: str,
        skip_logic: callable
    ) -> pd.DataFrame:
        """
        Legge un singolo CSV da locale o S3 applicando la skip_logic passata dal chiamante.
        """
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