import json, os, boto3
from urllib.parse import urlparse
from botocore.exceptions import ClientError

class LocalMetricsDAO:
    def save(self, path: str, metrics: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def load(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nessun file di metriche trovato in '{path}'.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

class S3MetricsDAO:
    def __init__(self):
        self.s3 = boto3.client("s3")

    def save(self, path: str, metrics: dict):
        parsed = urlparse(path)          # s3://bucket/key
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        self.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metrics, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

    def load(self, path: str) -> dict:
        parsed = urlparse(path)          # s3://bucket/key
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        try:
            resp = self.s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Nessun file di metriche trovato in 's3://{bucket}/{key}'.")
            raise
        return json.loads(resp["Body"].read().decode("utf-8"))

class MetricsDAOFactory:
    @staticmethod
    def get_dao(environment: str):
        return S3MetricsDAO() if environment == "aws" else LocalMetricsDAO()