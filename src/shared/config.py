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
            cls._instance.mode = os.getenv("TRAINING_MODE", "centralized")
            cls._instance.env = os.getenv("ENV_MODE", "local")

            # Validazione fondamentale per la robustezza del sistema
            if cls._instance.mode not in ["centralized", "federated"]:
                raise ValueError(f"TRAINING_MODE non valido nel file .env: {cls._instance.mode}")
            
            cls._instance.aws_region = os.getenv("AWS_REGION", "us-east-1")

            queue_prefix = os.getenv("SQS_QUEUE_PREFIX", "")
            queue_suffix = ".fifo" if cls._instance.env == "aws" else ""
            
            cls._instance.sqs_centralized_queue = f"{queue_prefix}centralized_queue{queue_suffix}"
            cls._instance.sqs_federated_queue = f"{queue_prefix}federated_queue{queue_suffix}"

            cls._instance.s3_bucket_name = os.getenv("DATASETS_BUCKET_NAME", "")
            if cls._instance.env == "aws" and not cls._instance.s3_bucket_name:
                raise ValueError("DATASETS_BUCKET_NAME non specificato nel file .env per l'ambiente AWS.")
            
            print(f"[CONFIG] Sistema caricato: {cls._instance.mode.upper()} | Ambiente: {cls._instance.env.upper()}")
            
        return cls._instance