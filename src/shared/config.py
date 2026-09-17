import os
import boto3
from dotenv import load_dotenv

class SystemConfig:
    """
    Configurazione di sistema, letta una sola volta dal file .env (pattern
    singleton: __new__ restituisce sempre la stessa istanza nello stesso
    processo) e condivisa da client, orchestrator e worker. Centralizza i
    due parametri che determinano il comportamento dell'intero sistema:

      - mode ('centralized'/'federated', da TRAINING_MODE): quale strategia
        di distribuzione usare;
      - env ('local'/'aws', da ENV_MODE): quale implementazione concreta dei
        servizi infrastrutturali (coda messaggi, storage dataset, database di
        stato) usare — mock locali su file, o i servizi AWS reali (SQS, S3,
        DynamoDB) tramite le rispettive factory.

    Deriva anche i nomi delle code SQS (con suffisso .fifo solo su AWS) e il
    nome del bucket S3 dei dataset, così questi dettagli di naming non sono
    duplicati in ogni componente che ne ha bisogno.
    """

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
            
            cls._instance.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

            queue_prefix = os.getenv("SQS_QUEUE_PREFIX", "")
            queue_suffix = ".fifo" if cls._instance.env == "aws" else ""
            
            cls._instance.sqs_centralized_queue = f"{queue_prefix}centralized_queue{queue_suffix}"
            cls._instance.sqs_federated_queue = f"{queue_prefix}federated_queue{queue_suffix}"

            cls._instance.s3_bucket_name = os.getenv("DATASETS_BUCKET_NAME", "")
            if cls._instance.env == "aws" and not cls._instance.s3_bucket_name:
                raise ValueError("DATASETS_BUCKET_NAME non specificato nel file .env per l'ambiente AWS.")
            
            print(f"[CONFIG] Sistema caricato: {cls._instance.mode.upper()} | Ambiente: {cls._instance.env.upper()}")
            
        return cls._instance