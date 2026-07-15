"""
Script di provisioning delle risorse AWS necessarie al sistema.
Va eseguito UNA VOLTA (o ogni volta che si vuole ricreare l'infrastruttura
da zero) prima di avviare main.py / gli orchestratori con SYS_ENV=aws.

Copre:
  1. Tabelle DynamoDB   -> ModelStatus, WorkerTasks, OrchestratorLocks,
                            workers_registry, orchestrators_registry
  2. Code SQS           -> centralized_queue, federated_queue
  3. Bucket S3          -> dataset + checkpoint dei modelli
  4. Launch Template + Auto Scaling Group per i nodi Worker EC2

Uso:
    pip install boto3 --break-system-packages
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    python setup_aws_resources.py

Le sezioni DynamoDB/SQS/S3 sono pronte "as-is". La sezione EC2/ASG richiede
di compilare le costanti in cima (AMI, subnet, security group, key pair,
IAM instance profile) coi valori del TUO account: sono valori specifici
dell'infrastruttura che non posso indovinare.
"""

import time
import boto3
from botocore.exceptions import ClientError

# =========================================================================
# CONFIGURAZIONE - modifica questi valori in base al tuo account/progetto
# =========================================================================

REGION = "eu-west-1"

S3_BUCKET_NAME = "distributed-rf-datasets"          # deve essere globalmente univoco su S3
SQS_QUEUE_NAMES = ["centralized_queue.fifo", "federated_queue.fifo"]

# --- Parametri EC2 / Auto Scaling Group (DA COMPILARE) ---
EC2_AMI_ID = "ami-XXXXXXXXXXXXXXXXX"                # es. Amazon Linux 2023 nella tua region
EC2_INSTANCE_TYPE = "t3.medium"
EC2_KEY_PAIR_NAME = "la-tua-keypair"                 # per SSH, opzionale se usi Session Manager
EC2_SECURITY_GROUP_IDS = ["sg-XXXXXXXXXXXXXXXXX"]
EC2_SUBNET_IDS = ["subnet-XXXXXXXXXXXXXXXXX", "subnet-YYYYYYYYYYYYYYYYY"]  # almeno 2 AZ per resilienza
EC2_IAM_INSTANCE_PROFILE = "distributed-rf-worker-profile"  # ruolo IAM con permessi su DynamoDB/SQS/S3
ASG_MIN_SIZE = 1
ASG_MAX_SIZE = 6
ASG_DESIRED_CAPACITY = 2

# User-data eseguito all'avvio di ogni worker EC2: clona/aggiorna il codice
# e avvia il processo worker. Personalizza il comando finale in base a come
# hai strutturato l'avvio dei worker nel tuo main.py / modulo worker.
WORKER_USER_DATA = f"""#!/bin/bash
set -e
export SYS_ENV=aws
export SYS_MODE=centralized
export AWS_REGION={REGION}
export S3_BUCKET_NAME={S3_BUCKET_NAME}
cd /opt/distributed-rf
# TODO: sostituisci con il comando reale di avvio del worker del tuo progetto
python3 -m src.worker.main
"""

# Inizializzazione Client AWS via Boto3
dynamodb = boto3.client("dynamodb", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
autoscaling = boto3.client("autoscaling", region_name=REGION)


# =========================================================================
# 1. DynamoDB
# =========================================================================

DYNAMO_TABLES = {
    "ModelStatus": "job_id",
    "WorkerTasks": "task_id",
    "OrchestratorLocks": "lock_key",
    "workers_registry": "worker_name",
    "orchestrators_registry": "orchestrator_name",
}


def create_dynamo_tables():
    for table_name, pk_name in DYNAMO_TABLES.items():
        try:
            dynamodb.create_table(
                TableName=table_name,
                KeySchema=[{"AttributeName": pk_name, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": pk_name, "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",  # on-demand: niente capacità da stimare a mano
            )
            print(f"[DynamoDB] Creazione tabella '{table_name}' avviata (PK: {pk_name}).")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUseException":
                print(f"[DynamoDB] Tabella '{table_name}' già esistente, salto.")
            else:
                raise

    # Attende che tutte le tabelle siano ACTIVE prima di proseguire
    for table_name in DYNAMO_TABLES:
        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
        print(f"[DynamoDB] Tabella '{table_name}' ACTIVE.")



# =========================================================================
# 2. SQS
# =========================================================================

def create_sqs_queues():
    for queue_name in SQS_QUEUE_NAMES:
        try:
            attributes = {
                "VisibilityTimeout": "60",       # coerente col default usato da BaseOrchestrator.start()
                "MessageRetentionPeriod": "86400",  # 1 giorno
            }
            
            # Se la coda è FIFO, AWS richiede esplicitamente l'attributo FifoQueue
            if queue_name.endswith(".fifo"):
                attributes["FifoQueue"] = "true"
            
            response = sqs.create_queue(
                QueueName=queue_name,
                Attributes=attributes,
            )
            print(f"[SQS] Coda '{queue_name}' creata: {response['QueueUrl']}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "QueueAlreadyExists":
                print(f"[SQS] Coda '{queue_name}' già esistente.")
            else:
                raise