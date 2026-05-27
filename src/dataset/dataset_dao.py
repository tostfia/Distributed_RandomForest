from abc import ABC, abstractmethod
import os
import pandas as pd
import io

class DatasetDAO(ABC):
    """Interfaccia astratta che definisce il contratto per l'accesso ai dati."""
    
    @abstractmethod
    def load_dataset(self, path: str) -> pd.DataFrame:
        """Carica un dataset e restituisce un DataFrame Pandas."""
        pass

    @abstractmethod
    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        """Salva un DataFrame Pandas nella destinazione specificata."""
        pass


class LocalFileSystemDAO(DatasetDAO):
    """Implementazione DAO per il File System Locale."""
    
    def load_dataset(self, path: str) -> pd.DataFrame:
        print(f"[DAO-LOCAL] Caricamento del dataset dal File System locale: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Il file locale {path} non esiste.")
        return pd.read_csv(path)

    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        print(f"[DAO-LOCAL] Salvataggio del dataset in locale su: {path}")
        # Crea le cartelle intermedie se non esistono
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


class AwsS3DAO(DatasetDAO):
    """Implementazione DAO per AWS S3 Storage."""
    
    def __init__(self):
        # Inizializziamo il client boto3 per S3
        import boto3
        self.s3_client = boto3.client('s3')

    def _parse_s3_uri(self, s3_uri: str):
        """Funzione di utilità per spezzare s3://bucket/path in (bucket, key)."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Formato URI S3 non valido: {s3_uri}. Deve iniziare con s3://")
        parts = s3_uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def load_dataset(self, path: str) -> pd.DataFrame:
        print(f"[DAO-AWS] Caricamento del dataset dal bucket S3: {path}")
        bucket, key = self._parse_s3_uri(path)
        
        # Scarichiamo l'oggetto da S3 direttamente in memoria come stream di byte
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        
        if status == 200:
            return pd.read_csv(io.BytesIO(response['Body'].read()))
        else:
            raise Exception(f"Errore nel download da S3. Status code: {status}")

    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        print(f"[DAO-AWS] Salvataggio del dataset nel bucket S3 su: {path}")
        bucket, key = self._parse_s3_uri(path)
        
        # Convertiamo il DataFrame in una stringa CSV in memoria
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Carichiamo il buffer su S3
        self.s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())