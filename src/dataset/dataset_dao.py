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

    @abstractmethod
    def save_binary(self, path: str, data: bytes) -> None:
        """Salva dati binari generici (es. modelli pickle) sulla destinazione."""
        pass
    @abstractmethod
    def exists(self, path: str) -> bool:
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

    def exists(self, path: str) -> bool:
        return os.path.exists(path)


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
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        # Carichiamo il buffer su S3
        s3_client = self._get_isolated_client()
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_data)
        print()  # Per andare a capo dopo la barra di avanzamento
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