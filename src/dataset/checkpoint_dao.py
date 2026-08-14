"""
CheckpointDAO
=============

Astrae la persistenza di oggetti Python serializzati (alberi parziali,
modello globale RandomForest, chunk di inferenza) in modo che il codice
degli Orchestratori non debba mai sapere se sta scrivendo su disco locale
o su S3.

Con questo DAO, chi chiama scrive semplicemente:

    dao = CheckpointDAOFactory.get_dao(self.environment)
    dao.save(path, obj)
    obj = dao.load(path)          # solleva FileNotFoundError se assente
    dao.exists(path)
    dao.delete(path)

e il path (locale o "s3://...") viene risolto correttamente in entrambi
i casi.
"""

import os
import pickle
from abc import ABC, abstractmethod
import boto3
from botocore.exceptions import ClientError


class CheckpointDAO(ABC):
    """Interfaccia comune per la persistenza dei checkpoint."""

    @abstractmethod
    def save(self, path: str, obj) -> None:
        """Serializza e salva `obj` in `path`, sovrascrivendo se già presente."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        """Carica e deserializza l'oggetto in `path`.

        Solleva FileNotFoundError se l'oggetto non esiste, in modo
        uniforme sia per il backend locale sia per quello S3.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str) -> None:
        """Cancella l'oggetto in `path`. Idempotente: non fallisce se assente."""
        raise NotImplementedError


class LocalCheckpointDAO(CheckpointDAO):
    """Backend per ambiente 'local' (Docker Compose / test in sviluppo).

    Scrive in modo quasi-atomico: prima su un file temporaneo nella stessa
    cartella, poi rinomina (`os.replace`), così un crash a metà scrittura
    non lascia mai un checkpoint corrotto/troncato sul disco.
    """

    def save(self, path: str, obj) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp-{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def delete(self, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)


class S3CheckpointDAO(CheckpointDAO):
    """Backend per ambiente 'aws'. Path atteso: 's3://bucket-name/prefix/key.pkl'."""

    def __init__(self):
        self._client = boto3.client("s3")
        self._ClientError = ClientError

    @staticmethod
    def _parse(path: str) -> tuple[str, str]:
        if not path.startswith("s3://"):
            raise ValueError(f"Path non in formato s3://: {path}")
        without_scheme = path[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise ValueError(f"Path S3 malformato (bucket o key mancante): {path}")
        return bucket, key

    def save(self, path: str, obj) -> None:
        bucket, key = self._parse(path)
        body = pickle.dumps(obj)
        self._client.put_object(Bucket=bucket, Key=key, Body=body)

    def load(self, path: str):
        bucket, key = self._parse(path)
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return pickle.loads(response["Body"].read())
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(path) from e
            raise

    def exists(self, path: str) -> bool:
        bucket, key = self._parse(path)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, path: str) -> None:
        bucket, key = self._parse(path)
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code not in ("404", "NoSuchKey"):
                raise


class CheckpointDAOFactory:
    """Factory a singleton, simmetrica a DatasetDAOFactory già usata nel progetto."""

    _local_instance: "LocalCheckpointDAO | None" = None
    _s3_instance: "S3CheckpointDAO | None" = None

    @classmethod
    def get_dao(cls, environment: str) -> CheckpointDAO:
        if environment == "aws":
            if cls._s3_instance is None:
                cls._s3_instance = S3CheckpointDAO()
            return cls._s3_instance

        if cls._local_instance is None:
            cls._local_instance = LocalCheckpointDAO()
        return cls._local_instance