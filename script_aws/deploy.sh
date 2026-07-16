#!/bin/bash
set -e

# =====================================================================
# DEPLOY: Random Forest Distribuito su ECS Fargate (AWS Academy Learner Lab)
# =====================================================================
# Prerequisiti:
#   - AWS CLI configurato con le credenziali del Learner Lab (~/.aws/credentials)
#   - Docker funzionante (docker --version)
#   - Da lanciare dalla root del progetto (dove sta il Dockerfile)
# =====================================================================

# ---------------------- VARIABILI DA CONTROLLARE UNA VOLTA -----------
REGION="us-east-1"
CLUSTER_NAME="forest-cluster"
REPO_NAME="rf-distributed"
SG_NAME="rf-distributed-sg"
RPC_PORT=18861

WORKER_DESIRED_COUNT=2
ORCHESTRATOR_DESIRED_COUNT=2

WORKER_CPU=1024
WORKER_MEMORY=2048
ORCH_CPU=2048
ORCH_MEMORY=4096

BUCKET_NAME="my-cluster-datasets-bucket-759804778194-us-east-1-an"

# TRAINING_MODE: parametrizzabile.
# Uso: ./deploy.sh                -> usa il default sotto (centralized)
#      ./deploy.sh federated      -> deploya in modalità federated
#      ./deploy.sh centralized    -> deploya in modalità centralized
TRAINING_MODE="${1:-centralized}"

if [[ "$TRAINING_MODE" != "centralized" && "$TRAINING_MODE" != "federated" ]]; then
  echo "ERRORE: TRAINING_MODE deve essere 'centralized' o 'federated', ricevuto: '$TRAINING_MODE'"
  exit 1
fi
# -----------------------------------------------------------------------

echo "==> [0/10] Controllo di sicurezza: .env non deve finire nell'immagine..."
if [ -f ".env" ] && [ ! -f ".dockerignore" ] || ( [ -f ".env" ] && ! grep -qxF ".env" .dockerignore 2>/dev/null ); then
  echo "    [ATTENZIONE] .env esiste ma non è escluso da .dockerignore."
  echo "    Aggiungo '.env' a .dockerignore per evitare di includere credenziali nell'immagine..."
  echo ".env" >> .dockerignore
fi
echo "    OK."

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

echo "==> [8/10] Registrazione Task Definition WORKER..."
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
        {"name": "RPC_PORT", "value": "${RPC_PORT}"},
        {"name": "ENV_MODE", "value": "aws"},
        {"name": "SYS_ENV", "value": "aws"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "SYS_MODE", "value": "${TRAINING_MODE}"},
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
echo "    Task Definition worker registrata."

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
        {"name": "ENV_MODE", "value": "aws"},
        {"name": "SYS_ENV", "value": "aws"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "SYS_MODE", "value": "${TRAINING_MODE}"},
        {"name": "EC2_ID", "value": "Fargate"},
        {"name": "RUNNING_IN_DOCKER", "value": "true"},
        {"name": "AWS_DEFAULT_REGION", "value": "${REGION}"},
        {"name": "DATASETS_BUCKET_NAME", "value": "${BUCKET_NAME}"}
      ],
      "command": [
        "sh", "-c",
        "export ORCHESTRATOR_ID=Orchestrator-\${EC2_ID}-\${TRAINING_MODE}-\$(hostname); exec python -m src.master.orchestrator.main"
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

# --- Worker service ---
WORKER_SVC_EXISTS=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services worker-service \
  --query "services[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

if [ "$WORKER_SVC_EXISTS" == "ACTIVE" ]; then
  echo "    Service worker-service esistente: aggiorno la task definition..."
  aws ecs update-service --cluster "$CLUSTER_NAME" --service worker-service \
    --task-definition rf-worker-task --desired-count "$WORKER_DESIRED_COUNT" \
    --region "$REGION" > /dev/null
else
  echo "    Creazione Service worker-service (desired-count=$WORKER_DESIRED_COUNT)..."
  aws ecs create-service \
    --cluster "$CLUSTER_NAME" \
    --service-name worker-service \
    --task-definition rf-worker-task \
    --desired-count "$WORKER_DESIRED_COUNT" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_CONFIG" \
    --region "$REGION" > /dev/null
fi

# --- Orchestrator service ---
ORCH_SVC_EXISTS=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services orchestrator-service \
  --query "services[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

if [ "$ORCH_SVC_EXISTS" == "ACTIVE" ]; then
  echo "    Service orchestrator-service esistente: aggiorno la task definition..."
  aws ecs update-service --cluster "$CLUSTER_NAME" --service orchestrator-service \
    --task-definition rf-orchestrator-task --desired-count "$ORCHESTRATOR_DESIRED_COUNT" \
    --region "$REGION" > /dev/null
else
  echo "    Creazione Service orchestrator-service (desired-count=$ORCHESTRATOR_DESIRED_COUNT)..."
  aws ecs create-service \
    --cluster "$CLUSTER_NAME" \
    --service-name orchestrator-service \
    --task-definition rf-orchestrator-task \
    --desired-count "$ORCHESTRATOR_DESIRED_COUNT" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_CONFIG" \
    --region "$REGION" > /dev/null
fi

echo ""
echo "========================================================================"
echo " DEPLOY COMPLETATO"
echo "========================================================================"
echo " Training mode:  $TRAINING_MODE"
echo " Cluster:        $CLUSTER_NAME"
echo " Security Group: $SG_ID"
echo " Subnet usate:   $SUBNET_1, $SUBNET_2"
echo " Log worker:      /ecs/rf-worker (CloudWatch)"
echo " Log orchestrator:/ecs/rf-orchestrator (CloudWatch)"
echo ""
echo " Per scalare i worker:"
echo "   aws ecs update-service --cluster $CLUSTER_NAME --service worker-service --desired-count N --region $REGION"
echo ""
echo " Per vedere i task attivi:"
echo "   aws ecs list-tasks --cluster $CLUSTER_NAME --region $REGION"
echo "========================================================================"