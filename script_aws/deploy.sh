#!/bin/bash
set -e

# =====================================================================
# DEPLOY: Random Forest Distribuito su ECS Fargate (AWS Academy Learner Lab)
# =====================================================================
# Prerequisiti:
#   - File .env presente nella root con le credenziali AWS aggiornate
#   - Docker funzionante (docker --version)
#   - Da lanciare dalla root del progetto (dove sta il Dockerfile)
# =====================================================================

# ---------------------------------------------------------------------
# [SETUP ENV-FIRST] Lettura e sincronizzazione dinamica da file .env
# ---------------------------------------------------------------------

ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
  echo "==> [PRE-CHECK] Caricamento configurazioni e credenziali da $ENV_FILE..."

  # Funzione helper per estrarre il valore di una chiave ignorando spazi attorno all'uguale e virgolette
  get_env_var() {
    local key="$1"
    grep -E "^[[:space:]]*${key}[[:space:]]*=" "$ENV_FILE" | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  }

  ENV_AWS_REGION=$(get_env_var "AWS_DEFAULT_REGION")
  if [ -n "$ENV_AWS_REGION" ]; then export AWS_DEFAULT_REGION="$ENV_AWS_REGION"; fi

  # 2. Caricamento Modalità di Training e Bucket S3
  ENV_TRAINING_MODE=$(get_env_var "TRAINING_MODE")
  ENV_BUCKET_NAME=$(get_env_var "DATASETS_BUCKET_NAME")
  ENV_NUM_WORKERS=$(get_env_var "NUM_WORKERS")
  ENV_DATASET_TYPE=$(get_env_var "DATASET_TYPE")
  ENV_RPC_SYNC_TIMEOUT=$(get_env_var "RPC_SYNC_TIMEOUT_SECONDS")
  ENV_RPC_INFERENCE_SYNC_TIMEOUT=$(get_env_var "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS")

  # Priorità per la modalità: parametro $1 > TRAINING_MODE > default "centralized"
  DETECTED_MODE="${1:-${ENV_TRAINING_MODE:-centralized}}"
  DETECTED_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  DETECTED_BUCKET="${ENV_BUCKET_NAME:-my-cluster-datasets-bucket-759804778194-us-east-1-an}"
  DETECTED_DATASET_TYPE="${ENV_DATASET_TYPE:-real}"
  DETECTED_WORKERS="${ENV_NUM_WORKERS:-2}"
  DETECTED_RPC_SYNC_TIMEOUT="${ENV_RPC_SYNC_TIMEOUT:-1800}"
  DETECTED_RPC_INFERENCE_SYNC_TIMEOUT="${ENV_RPC_INFERENCE_SYNC_TIMEOUT:-900}"
else
  echo "==> [ATTENZIONE] File $ENV_FILE non trovato. Uso parametri di fallback."
  DETECTED_MODE="${1:-centralized}"
  DETECTED_REGION="us-east-1"
  DETECTED_BUCKET="my-cluster-datasets-bucket-759804778194-us-east-1-an"
  DETECTED_WORKERS="2"
  DETECTED_DATASET_TYPE="real"
  DETECTED_RPC_SYNC_TIMEOUT="1800"
  DETECTED_RPC_INFERENCE_SYNC_TIMEOUT="900"
fi

REGION="$DETECTED_REGION"
CLUSTER_NAME="forest-cluster"
REPO_NAME="rf-distributed"
SG_NAME="rf-distributed-sg"
RPC_PORT=18861

WORKER_DESIRED_COUNT="$DETECTED_WORKERS"
ORCHESTRATOR_DESIRED_COUNT=2

# Timeout (in secondi) delle chiamate RPC sincrone che l'orchestratore fa verso
# i worker (vedi RPC_SYNC_TIMEOUT_SECONDS / RPC_INFERENCE_SYNC_TIMEOUT_SECONDS
# in centralized.py e federated.py). Solo l'orchestratore le usa: e' lui il lato
# chiamante della connessione rpyc, i worker non hanno bisogno di conoscerle.
RPC_SYNC_TIMEOUT_SECONDS="$DETECTED_RPC_SYNC_TIMEOUT"
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS="$DETECTED_RPC_INFERENCE_SYNC_TIMEOUT"

if ! [[ "$RPC_SYNC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || [ "$RPC_SYNC_TIMEOUT_SECONDS" -le 0 ]; then
  echo "ERRORE: RPC_SYNC_TIMEOUT_SECONDS deve essere un intero positivo (secondi), ricevuto: '$RPC_SYNC_TIMEOUT_SECONDS'"
  exit 1
fi
if ! [[ "$RPC_INFERENCE_SYNC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || [ "$RPC_INFERENCE_SYNC_TIMEOUT_SECONDS" -le 0 ]; then
  echo "ERRORE: RPC_INFERENCE_SYNC_TIMEOUT_SECONDS deve essere un intero positivo (secondi), ricevuto: '$RPC_INFERENCE_SYNC_TIMEOUT_SECONDS'"
  exit 1
fi

# Forziamo ECS a spegnere i task vecchi PRIMA di avviare quelli nuovi
# durante un deployment (niente overlap temporaneo vecchi+nuovi).
# Di default ECS userebbe minimumHealthyPercent=100/maximumPercent=200,
# che con desired-count=2 può far coesistere transitoriamente 4 task
# (2 vecchi + 2 nuovi) mentre partecipano entrambi alla leader election.
# Con 0/100 si accetta un breve downtime a favore di non avere mai più
# del desired-count di orchestrator/worker attivi in contemporanea.
DEPLOYMENT_CONFIG="minimumHealthyPercent=0,maximumPercent=100"

# Alzato da 1024/4096 (1 vCPU) a 4096/16384 (4 vCPU): un worker isolato su
# Fargate ha la sua CPU dedicata, quindi con piu' vCPU puo' davvero parallelizzare
# il pool di processi (vedi fix in BaseWorker.py) invece di restare a 1 processo
# per mancanza di core. Con 4 vCPU per worker il totale richiesto contemporaneamente
# sale a ~34 vCPU (7 worker x 4 = 28, + 2 orchestrator x 2 = 4, + eventuale
# test-engine = 2). Non verificabile in anticipo se rientra nella quota Fargate
# del Learner Lab (permesso negato su servicequotas): se il deploy fallisce con
# un errore di capacita' su create-service/run-task, abbassare questo valore
# (es. a 2048/8192) e ripetere.
WORKER_CPU=4096
WORKER_MEMORY=16384
ORCH_CPU=2048
ORCH_MEMORY=8192

BUCKET_NAME="$DETECTED_BUCKET"
TRAINING_MODE="$DETECTED_MODE"
DATASET_TYPE="$DETECTED_DATASET_TYPE"

if [[ "$TRAINING_MODE" != "centralized" && "$TRAINING_MODE" != "federated" ]]; then
  echo "ERRORE: TRAINING_MODE deve essere 'centralized' o 'federated', ricevuto: '$TRAINING_MODE'"
  exit 1
fi

echo "    [ENV CONFIG] REGION        : $REGION"
echo "    [ENV CONFIG] TRAINING_MODE : $TRAINING_MODE"
echo "    [ENV CONFIG] BUCKET_NAME   : $BUCKET_NAME"
echo "    [ENV CONFIG] NUM_WORKERS   : $WORKER_DESIRED_COUNT"
echo "    [ENV CONFIG] RPC_SYNC_TIMEOUT_SECONDS           : ${RPC_SYNC_TIMEOUT_SECONDS}s"
echo "    [ENV CONFIG] RPC_INFERENCE_SYNC_TIMEOUT_SECONDS : ${RPC_INFERENCE_SYNC_TIMEOUT_SECONDS}s"
echo "-----------------------------------------------------------------------"

echo "==> [0/10] Controllo di sicurezza: .env non deve finire nell'immagine..."
if [ -f ".env" ] && [ ! -f ".dockerignore" ] || ( [ -f ".env" ] && ! grep -qxF ".env" .dockerignore 2>/dev/null ); then
  echo "    [ATTENZIONE] .env esiste ma non è escluso da .dockerignore."
  echo "    Aggiungo '.env' a .dockerignore per evitare di includere credenziali nell'immagine..."
  echo ".env" >> .dockerignore
fi
echo "    OK."

echo "==> [0b/10] Verifica credenziali AWS in ~/.aws/credentials..."
AWS_CREDENTIALS_FILE="$HOME/.aws/credentials"
if [ ! -f "$AWS_CREDENTIALS_FILE" ]; then
  echo "    [ERRORE] $AWS_CREDENTIALS_FILE non trovato."
  echo "    Le credenziali AWS non vengono più lette da .env: genera il file eseguendo:"
  echo "      bash aws_creds.sh"
  echo "    e rilancia questo script."
  exit 1
fi
echo "    OK: $AWS_CREDENTIALS_FILE trovato (verrà usato il profilo [default])."

echo "==> [1/10] Verifica identità AWS..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
echo "    Account ID: $ACCOUNT_ID"

LABROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/LabRole"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> [2/10] Recupero VPC di default e subnet..."
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=is-default,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
  echo "    Nessuna VPC default trovata, prendo la prima disponibile..."
  VPC_ID=$(aws ec2 describe-vpcs --query "Vpcs[0].VpcId" --output text --region "$REGION")
fi
echo "    VPC: $VPC_ID"

SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query "Subnets[*].SubnetId" --output text --region "$REGION")
SUBNET_ARRAY=($SUBNET_IDS)
SUBNET_1=${SUBNET_ARRAY[0]}
SUBNET_2=${SUBNET_ARRAY[1]:-${SUBNET_ARRAY[0]}}
echo "    Subnet selezionate: $SUBNET_1 $SUBNET_2"

echo "==> [3/10] Creazione/verifica Security Group..."
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION" 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Orchestrator-Worker RPC + SSH/debug" \
    --vpc-id "$VPC_ID" \
    --query "GroupId" --output text --region "$REGION")
  echo "    Security Group creato: $SG_ID"

  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port "$RPC_PORT" \
    --source-group "$SG_ID" \
    --region "$REGION" > /dev/null
  echo "    Regola ingress self-referencing su porta $RPC_PORT aggiunta."
else
  echo "    Security Group già esistente: $SG_ID (riutilizzo)"
fi

echo "==> [4/10] Creazione/verifica repository ECR..."
if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" > /dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" > /dev/null
  echo "    Repository ECR creato: $REPO_NAME"
else
  echo "    Repository ECR già esistente: $REPO_NAME"
fi

echo "==> [5/10] Login Docker su ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$ECR_REGISTRY" > /dev/null
echo "    Login OK."

echo "==> [6/10] Build e push immagine Docker..."
docker build -t "$REPO_NAME" .
docker tag "$REPO_NAME:latest" "$ECR_REGISTRY/$REPO_NAME:latest"
docker push "$ECR_REGISTRY/$REPO_NAME:latest"
echo "    Immagine pushata: $ECR_REGISTRY/$REPO_NAME:latest"

echo "==> [7/10] Creazione/verifica cluster ECS..."
CLUSTER_STATUS=$(aws ecs describe-clusters --clusters "$CLUSTER_NAME" \
  --query "clusters[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

if [ "$CLUSTER_STATUS" != "ACTIVE" ]; then
  aws ecs create-cluster --cluster-name "$CLUSTER_NAME" --region "$REGION" > /dev/null
  echo "    Cluster creato: $CLUSTER_NAME"
else
  echo "    Cluster già attivo: $CLUSTER_NAME"
fi

if [ "$TRAINING_MODE" == "federated" ] && [ "$DATASET_TYPE" != "synthetic" ]; then
  echo "==> [7b/10] Verifica provisioning shard federati su S3..."
  MISSING_SHARD=0
  for i in $(seq 1 "$WORKER_DESIRED_COUNT"); do
    aws s3api head-object --bucket "$BUCKET_NAME" --key "federated_shards/worker_${i}/train_shard.csv" --region "$REGION" > /dev/null 2>&1 || MISSING_SHARD=1
  done
  if [ "$MISSING_SHARD" -eq 1 ]; then
    echo "    [ERRORE] Shard mancanti su S3 per uno o più worker (1..$WORKER_DESIRED_COUNT)."
    echo "    Esegui prima: python -m scripts.provision_federated_shards --num-workers $WORKER_DESIRED_COUNT"
    exit 1
  fi
  echo "    OK: shard presenti per tutti i $WORKER_DESIRED_COUNT worker."
elif [ "$TRAINING_MODE" == "federated" ]; then
  echo "==> [7b/10] Dataset SINTETICO rilevato: nessun controllo shard su S3 necessario"
  echo "    (ogni worker genera autonomamente il proprio shard con scikit-learn)."
fi

if [ "$TRAINING_MODE" == "federated" ]; then
  echo "==> [8/10] Registrazione Task Definition WORKER (una per ciascun indice fisso 1..$WORKER_DESIRED_COUNT)..."
  for i in $(seq 1 "$WORKER_DESIRED_COUNT"); do
    cat <<EOF > /tmp/worker-task-def-${i}.json
{
  "family": "rf-worker-task-${i}",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "${WORKER_CPU}",
  "memory": "${WORKER_MEMORY}",
  "taskRoleArn": "${LABROLE_ARN}",
  "executionRoleArn": "${LABROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "worker",
      "image": "${ECR_REGISTRY}/${REPO_NAME}:latest",
      "essential": true,
      "environment": [
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "WORKER_HEARTBEAT_TIMEOUT", "value": "120"},
        {"name": "NUM_WORKERS", "value": "${WORKER_DESIRED_COUNT}"},
        {"name": "WORKER_INDEX", "value": "${i}"},
        {"name": "RPC_PORT", "value": "${RPC_PORT}"},
        {"name": "ENV_MODE", "value": "aws"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "EC2_ID", "value": "Fargate"},
        {"name": "RUNNING_IN_DOCKER", "value": "true"},
        {"name": "AWS_DEFAULT_REGION", "value": "${REGION}"},
        {"name": "DATASETS_BUCKET_NAME", "value": "${BUCKET_NAME}"},
        {"name": "DATASET_TYPE", "value": "${DATASET_TYPE}"}
      ],
      "command": [
        "sh", "-c",
        "export RPC_ADVERTISE_HOST=\$(curl -s \"\$ECS_CONTAINER_METADATA_URI_V4\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['Networks'][0]['IPv4Addresses'][0])\"); echo \"Registrazione con IP: \$RPC_ADVERTISE_HOST (WORKER_INDEX=${i})\"; exec python -m src.worker.main Worker-\${EC2_ID}-\${TRAINING_MODE}-\$(hostname) ${RPC_PORT} ${TRAINING_MODE} aws"
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rf-worker",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "worker-${i}",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
EOF
    aws ecs register-task-definition --cli-input-json file:///tmp/worker-task-def-${i}.json --region "$REGION" > /dev/null
    echo "    Task Definition worker #$i registrata (rf-worker-task-${i}, WORKER_INDEX=${i})."
  done
else
  echo "==> [8/10] Registrazione Task Definition WORKER (unica, worker anonimi e intercambiabili)..."
  cat <<EOF > /tmp/worker-task-def.json
{
  "family": "rf-worker-task",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "${WORKER_CPU}",
  "memory": "${WORKER_MEMORY}",
  "taskRoleArn": "${LABROLE_ARN}",
  "executionRoleArn": "${LABROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "worker",
      "image": "${ECR_REGISTRY}/${REPO_NAME}:latest",
      "essential": true,
      "environment": [
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "WORKER_HEARTBEAT_TIMEOUT", "value": "120"},
        {"name": "NUM_WORKERS", "value": "${WORKER_DESIRED_COUNT}"},
        {"name": "RPC_PORT", "value": "${RPC_PORT}"},
        {"name": "ENV_MODE", "value": "aws"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "EC2_ID", "value": "Fargate"},
        {"name": "RUNNING_IN_DOCKER", "value": "true"},
        {"name": "AWS_DEFAULT_REGION", "value": "${REGION}"},
        {"name": "DATASETS_BUCKET_NAME", "value": "${BUCKET_NAME}"}
      ],
      "command": [
        "sh", "-c",
        "export RPC_ADVERTISE_HOST=\$(curl -s \"\$ECS_CONTAINER_METADATA_URI_V4\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['Networks'][0]['IPv4Addresses'][0])\"); echo \"Registrazione con IP: \$RPC_ADVERTISE_HOST\"; exec python -m src.worker.main Worker-\${EC2_ID}-\${TRAINING_MODE}-\$(hostname) ${RPC_PORT} ${TRAINING_MODE} aws"
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rf-worker",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "worker",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
EOF
  aws ecs register-task-definition --cli-input-json file:///tmp/worker-task-def.json --region "$REGION" > /dev/null
  echo "    Task Definition worker registrata (rf-worker-task, desired-count=$WORKER_DESIRED_COUNT anonimi)."
fi

echo "==> [9/10] Registrazione Task Definition ORCHESTRATOR..."
cat <<EOF > /tmp/orchestrator-task-def.json
{
  "family": "rf-orchestrator-task",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "${ORCH_CPU}",
  "memory": "${ORCH_MEMORY}",
  "taskRoleArn": "${LABROLE_ARN}",
  "executionRoleArn": "${LABROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "orchestrator",
      "image": "${ECR_REGISTRY}/${REPO_NAME}:latest",
      "essential": true,
      "environment": [
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "WORKER_HEARTBEAT_TIMEOUT", "value": "120"},
        {"name": "NUM_WORKERS", "value": "${WORKER_DESIRED_COUNT}"},
        {"name": "ENV_MODE", "value": "aws"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "EC2_ID", "value": "Fargate"},
        {"name": "RUNNING_IN_DOCKER", "value": "true"},
        {"name": "AWS_DEFAULT_REGION", "value": "${REGION}"},
        {"name": "DATASETS_BUCKET_NAME", "value": "${BUCKET_NAME}"},
        {"name": "RPC_SYNC_TIMEOUT_SECONDS", "value": "${RPC_SYNC_TIMEOUT_SECONDS}"},
        {"name": "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", "value": "${RPC_INFERENCE_SYNC_TIMEOUT_SECONDS}"}
      ],
      "command": [
        "sh", "-c",
        "export ORCHESTRATOR_INDEX=\$(curl -s \"\$ECS_CONTAINER_METADATA_URI_V4/task\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['TaskARN'].split('/')[-1])\"); echo \"Registrazione con Task ID: \$ORCHESTRATOR_INDEX\"; exec python -m src.master.orchestrator.main"
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rf-orchestrator",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "orchestrator",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
EOF
aws ecs register-task-definition --cli-input-json file:///tmp/orchestrator-task-def.json --region "$REGION" > /dev/null
echo "    Task Definition orchestrator registrata."

echo "==> [10/10] Creazione/aggiornamento Services..."

NETWORK_CONFIG="awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"

if [ "$TRAINING_MODE" == "federated" ]; then
  # --- Worker services: uno per ciascun indice fisso, desired-count=1 ciascuno ---
  for i in $(seq 1 "$WORKER_DESIRED_COUNT"); do
    SVC_NAME="worker-service-${i}"
    SVC_EXISTS=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$SVC_NAME" \
      --query "services[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

    if [ "$SVC_EXISTS" == "ACTIVE" ]; then
      echo "    Service $SVC_NAME esistente: forzo il rimpiazzo del container (WORKER_INDEX=${i})..."
      aws ecs update-service --cluster "$CLUSTER_NAME" --service "$SVC_NAME" \
        --task-definition "rf-worker-task-${i}" --desired-count 1 \
        --deployment-configuration "$DEPLOYMENT_CONFIG" \
        --availability-zone-rebalancing DISABLED \
        --force-new-deployment --region "$REGION" > /dev/null
    else
      echo "    Creazione Service $SVC_NAME (WORKER_INDEX=${i}, desired-count=1)..."
      aws ecs create-service \
        --cluster "$CLUSTER_NAME" \
        --service-name "$SVC_NAME" \
        --task-definition "rf-worker-task-${i}" \
        --desired-count 1 \
        --launch-type FARGATE \
        --deployment-configuration "$DEPLOYMENT_CONFIG" \
        --availability-zone-rebalancing DISABLED \
        --network-configuration "$NETWORK_CONFIG" \
        --region "$REGION" > /dev/null
    fi
  done
else
  # --- Worker service: unico, worker anonimi e intercambiabili, desired-count=N ---
  WORKER_SVC_EXISTS=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services worker-service \
    --query "services[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

  if [ "$WORKER_SVC_EXISTS" == "ACTIVE" ]; then
    echo "    Service worker-service esistente: forzo il rimpiazzo dei container..."
    aws ecs update-service --cluster "$CLUSTER_NAME" --service worker-service \
      --task-definition rf-worker-task --desired-count "$WORKER_DESIRED_COUNT" \
      --deployment-configuration "$DEPLOYMENT_CONFIG" \
      --availability-zone-rebalancing DISABLED \
      --force-new-deployment --region "$REGION" > /dev/null
  else
    echo "    Creazione Service worker-service (desired-count=$WORKER_DESIRED_COUNT)..."
    aws ecs create-service \
      --cluster "$CLUSTER_NAME" \
      --service-name worker-service \
      --task-definition rf-worker-task \
      --desired-count "$WORKER_DESIRED_COUNT" \
      --launch-type FARGATE \
      --deployment-configuration "$DEPLOYMENT_CONFIG" \
      --availability-zone-rebalancing DISABLED \
      --network-configuration "$NETWORK_CONFIG" \
      --region "$REGION" > /dev/null
  fi
fi

# --- Orchestrator service ---
ORCH_SVC_EXISTS=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services orchestrator-service \
  --query "services[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

if [ "$ORCH_SVC_EXISTS" == "ACTIVE" ]; then
  echo "    Service orchestrator-service esistente: forzo il rimpiazzo dei container..."
  aws ecs update-service --cluster "$CLUSTER_NAME" --service orchestrator-service \
    --task-definition rf-orchestrator-task --desired-count "$ORCHESTRATOR_DESIRED_COUNT" \
    --deployment-configuration "$DEPLOYMENT_CONFIG" \
    --availability-zone-rebalancing DISABLED \
    --force-new-deployment --region "$REGION" > /dev/null
else
  echo "    Creazione Service orchestrator-service (desired-count=$ORCHESTRATOR_DESIRED_COUNT)..."
  aws ecs create-service \
    --cluster "$CLUSTER_NAME" \
    --service-name orchestrator-service \
    --task-definition rf-orchestrator-task \
    --desired-count "$ORCHESTRATOR_DESIRED_COUNT" \
    --launch-type FARGATE \
    --deployment-configuration "$DEPLOYMENT_CONFIG" \
    --availability-zone-rebalancing DISABLED \
    --network-configuration "$NETWORK_CONFIG" \
    --region "$REGION" > /dev/null
fi

echo ""
echo "========================================================================"
echo " DEPLOY COMPLETATO"
echo "========================================================================"
echo " Training mode:  $TRAINING_MODE"
echo " Region:         $REGION"
echo " Bucket S3:      $BUCKET_NAME"
echo " Cluster:        $CLUSTER_NAME"
echo " Security Group: $SG_ID"
echo " Subnet usate:   $SUBNET_1, $SUBNET_2"
echo " RPC sync timeout (training/inferenza): ${RPC_SYNC_TIMEOUT_SECONDS}s / ${RPC_INFERENCE_SYNC_TIMEOUT_SECONDS}s"
echo " Log worker:      /ecs/rf-worker (CloudWatch)"
echo " Log orchestrator:/ecs/rf-orchestrator (CloudWatch)"
echo ""
if [ "$TRAINING_MODE" == "federated" ]; then
  echo " Per scalare i worker: aggiorna NUM_WORKERS in .env, ri-esegui"
  echo "   'python -m scripts.provision_federated_shards --num-workers N' e poi questo deploy.sh"
else
  echo " Per scalare i worker: aws ecs update-service --cluster $CLUSTER_NAME \\"
  echo "   --service worker-service --desired-count N --region $REGION"
fi
echo ""
echo " Per vedere i task attivi:"
echo "   aws ecs list-tasks --cluster $CLUSTER_NAME --region $REGION"
echo "========================================================================"