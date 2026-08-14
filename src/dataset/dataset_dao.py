from abc import ABC, abstractmethod
import os

import pandas as pd
import io
import boto3
from botocore.config import Config


class DatasetDAO(ABC):
    """Interfaccia astratta che definisce il contratto per l'accesso ai dati."""
    
    @abstractmethod
    def load_dataset(self, path: str, sample_fraction: float = None, dataset_seed: int = None) -> pd.DataFrame:
        """Carica un dataset e restituisce un DataFrame Pandas.

        Se sample_fraction è specificato (0 < f < 1), il campionamento avviene
        IN STREAMING durante la lettura, chunk-per-chunk, evitando di caricare
        l'intero file in RAM prima di sottocampionare. Fondamentale per dataset
        grandi (es. CICIDS2018) su ambienti con memoria limitata (Fargate).
        """
        pass

    @abstractmethod
    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        """Salva un DataFrame Pandas nella destinazione specificata."""
        pass

    @abstractmethod
    def save_binary(self, path: str, data: bytes) -> None:
        """Salva dati binari generici (es. modelli pickle) sulla destinazione."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass


class LocalFileSystemDAO(DatasetDAO):
    """Implementazione DAO per il File System Locale."""

    CHUNK_SIZE = 50000

    def load_dataset(self, path: str, sample_fraction: float = None, dataset_seed: int = None) -> pd.DataFrame:
        print(f"[DAO-LOCAL] Caricamento del dataset dal File System locale: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Il file locale {path} non esiste.")

        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            print(f"[DAO-LOCAL] Campionamento in streaming (frac={sample_fraction}) durante la lettura...")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(path, chunksize=self.CHUNK_SIZE, dtype=str, low_memory=False)):
                chunk_seed = (dataset_seed + i) if dataset_seed is not None else None
                chunks.append(chunk.sample(frac=sample_fraction, random_state=chunk_seed))
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            print(f"[DAO-LOCAL] [OK] Campionamento streaming completato: {df.shape[0]} righe mantenute.")
            return df

        return pd.read_csv(path)

    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        print(f"[DAO-LOCAL] Salvataggio del dataset in locale su: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def save_binary(self, path: str, data: bytes) -> None:
        print(f"[DAO-LOCAL] Salvataggio file binario su: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)


class AwsS3DAO(DatasetDAO):
    """Implementazione DAO per AWS S3 Storage."""

    CHUNK_SIZE = 50000

    def __init__(self):
        pass

    def _get_isolated_client(self):
        """Genera un client S3 isolato e specifico per il thread corrente, con timeout configurati."""

        local_session = boto3.Session()
        config_timeout = Config(
            connect_timeout=15,
            read_timeout=30,
            retries={'max_attempts': 5, "mode": "adaptive"}
        )
        return local_session.client('s3', config=config_timeout)

    def _parse_s3_uri(self, s3_uri: str):
        """Funzione di utilità per spezzare s3://bucket/path in (bucket, key)."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Formato URI S3 non valido: {s3_uri}. Deve iniziare con s3://")
        parts = s3_uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def load_dataset(self, path: str, sample_fraction: float = None, dataset_seed: int = None) -> pd.DataFrame:
        print(f"[DAO-AWS] Caricamento del dataset dal bucket S3: {path}")
        bucket, key = self._parse_s3_uri(path)

        s3_client = self._get_isolated_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status != 200:
            raise Exception(f"Errore nel download da S3. Status code: {status}")

        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            print(f"[DAO-AWS] Campionamento in streaming (frac={sample_fraction}) durante la lettura, "
                  f"evito di caricare l'intero file in RAM...")
            chunks = []
            # response['Body'] è uno StreamingBody: pandas lo consuma progressivamente,
            # senza mai materializzare l'intero oggetto S3 in memoria.
            reader = pd.read_csv(response['Body'], chunksize=self.CHUNK_SIZE, dtype=str, low_memory=False)
            for i, chunk in enumerate(reader):
                chunk_seed = (dataset_seed + i) if dataset_seed is not None else None
                chunks.append(chunk.sample(frac=sample_fraction, random_state=chunk_seed))
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            print(f"[DAO-AWS] [OK] Campionamento streaming completato: {df.shape[0]} righe mantenute.")
            return df

        return pd.read_csv(io.BytesIO(response['Body'].read()))

    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        print(f"[DAO-AWS] Salvataggio del dataset nel bucket S3 su: {path}")
        bucket, key = self._parse_s3_uri(path)

        csv_data = df.to_csv(index=False).encode('utf-8')

        s3_client = self._get_isolated_client()
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_data)
        print()
        print(f"[DAO-AWS] [OK] Salvataggio su S3 completato con successo!")

    def save_binary(self, path: str, data: bytes) -> None:
        print(f"[DAO-AWS] Salvataggio file binario nel bucket S3 su: {path}")
        bucket, key = self._parse_s3_uri(path)

        s3_client = self._get_isolated_client()
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        print(f"[DAO-AWS] [OK] Salvataggio binario completato con successo!")

    def exists(self, path: str) -> bool:
        bucket, key = self._parse_s3_uri(path)
        s3_client = self._get_isolated_client()
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except s3_client.exceptions.ClientError:
            return False