import os
import glob
import random
from typing import List, Sequence, Union

import pandas as pd

from src.shared.utilities.loader.datasetLoader import DatasetLoader
from src.dataset.dataset_dao import AwsS3DAO
from src.dataset.dataset_dao_factory import DatasetDAOFactory


class RawCSVDataLoader(DatasetLoader):
    """
    Loader CSV grezzo per sorgenti locali o S3.
    Delega l'accesso ai dati al DAO Factory di sistema per supportare
    sia l'esecuzione locale che l'ambiente distribuito AWS Cloud.
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

        # Inizializziamo il DAO corretto in base al SystemConfig()
        self.dao = DatasetDAOFactory.get_dao()

        self._validate_parameters()

    def load(self) -> pd.DataFrame:
        sources = self._discover_sources()
        chunks = []

        print("Caricamento dataset CSV grezzo...")
        print(f" • Sorgenti trovate: {len(sources)}")
        print(f" • Sample fraction:  {self.sample_fraction}")
        print(f" • Dataset seed:     {self.dataset_seed}")

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
        Determina la lista di sorgenti ordinata alfabeticamente.
        """
        if isinstance(self.data_url, (list, tuple)):
            sources = sorted(list(self.data_url))

        elif isinstance(self.data_url, str) and self._is_s3_path(self.data_url):
            if self.data_url.endswith("/"):
                print(f"[S3 DISCOVERY] Scansione directory Cloud: {self.data_url}")
                sources = self._list_s3_directory(self.data_url)
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

        return sources

    def _list_s3_directory(self, s3_dir_uri: str) -> List[str]:
        """Elenca i file .csv in una cartella S3 delegando la gestione delle credenziali al DAO."""
        clean_url = s3_dir_uri.replace("s3://", "")
        parts = clean_url.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # Otteniamo il client S3 delegandolo all'istanza AwsS3DAO se disponibile
        if isinstance(self.dao, AwsS3DAO):
            s3_client = self.dao._get_isolated_client()
        else:
            # Fallback sicuro con AwsS3DAO temporaneo per garantire le giuste credenziali/sessioni AWS
            s3_client = AwsS3DAO()._get_isolated_client()

        paginator = s3_client.get_paginator("list_objects_v2")
        sources = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".csv"):
                    sources.append(f"s3://{bucket}/{key}")

        return sorted(sources)

    def _read_single_csv(self, source: str) -> pd.DataFrame:
        try:
            # 1. Lettura delegata al DAO (gestisce automaticamente sia file locale che S3)
            df_temp = self.dao.load_dataset(source)

            # 2. Standardizzazione colonne
            df_temp.columns = [c.strip() for c in df_temp.columns]

            # 3. Uniformiamo il nome del target
            if "label" in df_temp.columns:
                df_temp = df_temp.rename(columns={"label": "Label"})

            cols_to_convert = df_temp.columns.difference(["Label"])
            df_temp[cols_to_convert] = df_temp[cols_to_convert].apply(pd.to_numeric, errors="coerce")

            # 4. Campionamento in memoria
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