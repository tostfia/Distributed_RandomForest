
import json, os, boto3
from urllib.parse import urlparse

class LocalMetricsDAO:
    def save(self, path: str, metrics: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

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

class MetricsDAOFactory:
    @staticmethod
    def get_dao(environment: str):
        return S3MetricsDAO() if environment == "aws" else LocalMetricsDAO()