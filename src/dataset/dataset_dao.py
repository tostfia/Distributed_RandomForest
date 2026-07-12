from abc import ABC, abstractmethod
import os
import sys
import pandas as pd
import io
import boto3
import logging
import threading



class _UploadProgress:
    """Callable che boto3 richiama periodicamente durante upload_fileobj,
    passando il numero di byte trasmessi in quel 'chunk'. Tiene il conteggio
    cumulativo ed emette una barra di avanzamento a riga singola."""
 
    def __init__(self, total_bytes: int, label: str = "Upload"):
        self._total_bytes = total_bytes
        self._label = label
        self._seen_so_far = 0
        self._lock = threading.Lock()
 
    def __call__(self, bytes_amount: int):
        with self._lock:
            self._seen_so_far += bytes_amount
            if self._total_bytes > 0:
                percentage = (self._seen_so_far / self._total_bytes) * 100
                mb_seen = self._seen_so_far / (1024 ** 2)
                mb_total = self._total_bytes / (1024 ** 2)
                sys.stdout.write(
                    f"\r[DAO-AWS] {self._label}: {mb_seen:6.1f} / {mb_total:6.1f} MB "
                    f"({percentage:5.1f}%)"
                )
            else:
                mb_seen = self._seen_so_far / (1024 ** 2)
                sys.stdout.write(f"\r[DAO-AWS] {self._label}: {mb_seen:6.1f} MB trasferiti")
            sys.stdout.flush()
            
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

    @abstractmethod
    def save_binary(self, path: str, data: bytes) -> None:
        """Salva dati binari generici (es. modelli pickle) sulla destinazione."""
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
    
    def save_binary(self, path: str, data: bytes) -> None:
        print(f"[DAO-LOCAL] Salvataggio file binario su: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


class AwsS3DAO(DatasetDAO):
    """Implementazione DAO per AWS S3 Storage."""
    
    def __init__(self):
        pass

    def _get_isolated_client(self):
        """Genera un client S3 isolato e specifico per il thread corrente, con timeout configurati."""
        import boto3
        from botocore.config import Config
        
        # Forza la creazione di una sessione pulita
        local_session = boto3.Session()
        
        # Configura i timeout: se la rete si incastra, l'operazione fallisce anziché freezare
        config_timeout = Config(
            connect_timeout=15,  # 15 secondi per connettersi
            read_timeout=30,     # 30 secondi per trasmettere i dati
            retries={'max_attempts': 2}
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

    def load_dataset(self, path: str) -> pd.DataFrame:
        print(f"[DAO-AWS] Caricamento del dataset dal bucket S3: {path}")
        bucket, key = self._parse_s3_uri(path)
        
        s3_client = self._get_isolated_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        
        if status == 200:
            return pd.read_csv(io.BytesIO(response['Body'].read()))
        else:
            raise Exception(f"Errore nel download da S3. Status code: {status}")
        
    def save_dataset(self, path: str, df: pd.DataFrame) -> None:
        print(f"[DAO-AWS] Salvataggio del dataset nel bucket S3 su: {path}")
        bucket, key = self._parse_s3_uri(path)
        
        # Convertiamo il DataFrame in una stringa CSV in memoria
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        total_bytes = csv_buffer.tell()
        csv_buffer.seek(0)
        
        # Carichiamo il buffer su S3
        s3_client = self._get_isolated_client()
        progress = _UploadProgress(total_bytes, label="Salvataggio dataset")
        s3_client.upload_fileobj(Fileobj=csv_buffer, Bucket=bucket, Key=key, Callback=progress)
        print()  # Per andare a capo dopo la barra di avanzamento
        print(f"[DAO-AWS] [OK] Salvataggio su S3 completato con successo!")
        
    def save_binary(self, path: str, data: bytes) -> None:
        print(f"[DAO-AWS] Salvataggio file binario nel bucket S3 su: {path}")
        bucket, key = self._parse_s3_uri(path)
        s3_client = self._get_isolated_client()
        buffer = io.BytesIO(data)
        total_bytes = len(data)
        progress = _UploadProgress(total_bytes, label="Salvataggio file binario")
        s3_client.upload_fileobj(Fileobj=buffer, Bucket=bucket, Key=key, Callback=progress)
        print()  # Per andare a capo dopo la barra di avanzamento
        print(f"[DAO-AWS] [OK] Salvataggio binario completato con successo!")