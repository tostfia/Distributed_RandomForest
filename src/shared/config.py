import os
from dotenv import load_dotenv

class SystemConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Carica le variabili dal file .env automaticamente
            load_dotenv() 
            
            cls._instance = super(SystemConfig, cls).__new__(cls)
            
            # Legge le variabili con i tuoi valori di default
            cls._instance.mode = os.getenv("SYS_MODE", "centralized")
            cls._instance.env = os.getenv("SYS_ENV", "local")
            
            # Validazione fondamentale per la robustezza del sistema
            if cls._instance.mode not in ["centralized", "federated"]:
                raise ValueError(f"SYS_MODE non valido nel file .env: {cls._instance.mode}")
            
            cls.instance.aws_region = os.getenv("AWS_REGION", "us-east-1")

            queue_prefix = os.getenv("SQS_QUEUE_PREFIX","")
            cls.instance.sqs_centralized_queue = f"{queue_prefix}centralized-queue"
            cls.instance.sqs_federated_queue = f"{queue_prefix}federated-queue"

            cls._instance.s3_bucket_name = os.getenv("S3_BUCKET_NAME","")
            if cls._instance.env == "aws" and not cls._instance.s3_bucket_name:
                raise ValueError("S3_BUCKET_NAME non specificato nel file .env per l'ambiente AWS.")
            
            print(f"[CONFIG] Sistema caricato: {cls._instance.mode.upper()} | Ambiente: {cls._instance.env.upper()}")
            
        return cls._instance