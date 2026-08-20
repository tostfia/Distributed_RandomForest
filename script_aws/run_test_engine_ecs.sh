#!/bin/bash
set -e

# =====================================================================
# RUN TEST ENGINE (ECS Exec): lancia il TestEngine DENTRO la VPC come
# task ECS one-off (non un service persistente) e ci entra con una shell
# interattiva via 'aws ecs execute-command' — stessa identica esperienza
# del prompt "Scelta (1-7, o 'all')" che hai in locale, ma con i worker
# raggiungibili sul loro IP privato (nessuna esposizione RPC su internet).
#
# Prerequisiti:
#   - deploy.sh già eseguito con successo (worker-service/orchestrator-service
#     RUNNING sul cluster 'forest-cluster')
#   - .env con SYS_ENV=aws e credenziali AWS Academy correnti
#
# Uso:
#   ./run_test_engine_ecs.sh
# =====================================================================

if [ ! -f .env ]; then
  echo "[ERRORE] File .env non trovato nella directory corrente ($(pwd))."
  exit 1
fi

get_env_var() {
  local key="$1"
  grep -E "^[[:space:]]*${key}[[:space:]]*=" .env | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}

ENV_SYS_ENV=$(get_env_var "SYS_ENV")
ENV_SYS_MODE=$(get_env_var "SYS_MODE")
ENV_TRAINING_MODE=$(get_env_var "TRAINING_MODE")
ENV_NUM_WORKERS=$(get_env_var "NUM_WORKERS")
ENV_REGION=$(get_env_var "AWS_DEFAULT_REGION")
ENV_BUCKET_NAME=$(get_env_var "DATASETS_BUCKET_NAME")

if [ "$ENV_SYS_ENV" != "aws" ]; then
  echo "[ERRORE] SYS_ENV nel .env è '$ENV_SYS_ENV', non 'aws'."
  exit 1
fi

TRAINING_MODE="${ENV_SYS_MODE:-${ENV_TRAINING_MODE:-centralized}}"
REGION="${ENV_REGION:-us-east-1}"
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"
BUCKET_NAME="${ENV_BUCKET_NAME:-my-cluster-datasets-bucket-759804778194-us-east-1-an}"
CLUSTER_NAME="forest-cluster"
REPO_NAME="rf-distributed"
SG_NAME="rf-distributed-sg"
FAMILY="rf-test-engine-task"
CONTAINER_NAME="test-engine"

if [[ "$TRAINING_MODE" != "centralized" && "$TRAINING_MODE" != "federated" ]]; then
  echo "ERRORE: SYS_MODE/TRAINING_MODE deve essere 'centralized' o 'federated', ricevuto: '$TRAINING_MODE'"
  exit 1
fi

echo "===================================================================="
echo " RUN TEST ENGINE (ECS Exec)  -  forest-cluster ($REGION)"
echo "===================================================================="
echo " TRAINING_MODE : $TRAINING_MODE"
echo " NUM_WORKERS   : $NUM_WORKERS"
echo "--------------------------------------------------------------------"

echo "==> [1/6] Verifica credenziali AWS (Learner Lab, scadono ogni ~4h)..."
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute. Aggiornale nel .env e riprova."
  exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
LABROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/LabRole"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo "    OK, credenziali valide."

echo "==> [2/6] Verifica che worker-service/orchestrator-service siano stabili..."
ALL_SERVICE_ARNS=$(aws ecs list-services --cluster "$CLUSTER_NAME" --region "$REGION" \
  --query "serviceArns[]" --output text 2>/dev/null || echo "")
TARGET_SERVICES=()
for arn in $ALL_SERVICE_ARNS; do
  svc_name="${arn##*/}"
  if [[ "$svc_name" == worker-service* || "$svc_name" == "orchestrator-service" ]]; then
    TARGET_SERVICES+=("$svc_name")
  fi
done
if [ "${#TARGET_SERVICES[@]}" -eq 0 ]; then
  echo "[ERRORE] Nessun Service worker/orchestrator trovato sul cluster '$CLUSTER_NAME'. Hai lanciato deploy.sh?"
  exit 1
fi
echo "    Service trovati: ${TARGET_SERVICES[*]}"
aws ecs wait services-stable --cluster "$CLUSTER_NAME" --services "${TARGET_SERVICES[@]}" --region "$REGION"
echo "    OK, infrastruttura pronta."

echo "==> [3/6] Recupero VPC/subnet/Security Group (stessi usati da deploy.sh)..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query "Subnets[*].SubnetId" --output text --region "$REGION")
SUBNET_ARRAY=($SUBNET_IDS)
SUBNET_1=${SUBNET_ARRAY[0]}
SUBNET_2=${SUBNET_ARRAY[1]:-${SUBNET_ARRAY[0]}}
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION")
if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
  echo "[ERRORE] Security Group '$SG_NAME' non trovato: hai lanciato deploy.sh?"
  exit 1
fi
echo "    VPC: $VPC_ID | Subnet: $SUBNET_1, $SUBNET_2 | SG: $SG_ID"
echo "    (Stesso SG dei worker: la regola self-referencing sulla porta 18861"
echo "     già presente basta, il test-engine sarà raggiungibile dai worker e"
echo "     viceversa senza aprire nulla verso l'esterno.)"

echo "==> [4/6] Registrazione Task Definition '$FAMILY'..."
# Il container resta in attesa ('sleep infinity'): l'engine NON parte da solo
# all'avvio del task, altrimenti il prompt interattivo 'input()' bloccherebbe
# il processo principale senza nessuno collegato a leggerlo. Ci entriamo noi
# dopo, con 'ecs execute-command', ed è lì che lanciamo davvero l'engine.
cat <<EOF > /tmp/test-engine-task-def.json
{
  "family": "${FAMILY}",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "2048",
  "memory": "8192",
  "taskRoleArn": "${LABROLE_ARN}",
  "executionRoleArn": "${LABROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "${CONTAINER_NAME}",
      "image": "${ECR_REGISTRY}/${REPO_NAME}:latest",
      "essential": true,
      "environment": [
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "NUM_WORKERS", "value": "${NUM_WORKERS}"},
        {"name": "ENV_MODE", "value": "aws"},
        {"name": "SYS_ENV", "value": "aws"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "SYS_MODE", "value": "${TRAINING_MODE}"},
        {"name": "EC2_ID", "value": "Fargate"},
        {"name": "RUNNING_IN_DOCKER", "value": "true"},
        {"name": "AWS_DEFAULT_REGION", "value": "${REGION}"},
        {"name": "DATASETS_BUCKET_NAME", "value": "${BUCKET_NAME}"}
      ],
      "command": ["sh", "-c", "sleep infinity"],
      "linuxParameters": {"initProcessEnabled": true},
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rf-test-engine",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "test-engine",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
EOF
aws ecs register-task-definition --cli-input-json file:///tmp/test-engine-task-def.json --region "$REGION" > /dev/null
echo "    Task Definition registrata."

echo "==> [5/6] Avvio del task (run-task, one-off, --enable-execute-command)..."
TASK_ARN=$(aws ecs run-task \
  --cluster "$CLUSTER_NAME" \
  --task-definition "$FAMILY" \
  --launch-type FARGATE \
  --enable-execute-command \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region "$REGION" \
  --query "tasks[0].taskArn" --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" == "None" ]; then
  echo "[ERRORE] run-task non ha restituito un Task ARN valido."
  exit 1
fi
echo "    Task avviato: $TASK_ARN"

echo "    Attendo che il task sia RUNNING e l'agente ECS Exec sia pronto..."
for attempt in $(seq 1 30); do
  TASK_STATUS=$(aws ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$TASK_ARN" \
    --query "tasks[0].lastStatus" --output text --region "$REGION")
  AGENT_STATUS=$(aws ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$TASK_ARN" \
    --query "tasks[0].containers[0].managedAgents[?name=='ExecuteCommandAgent'].lastStatus | [0]" \
    --output text --region "$REGION" 2>/dev/null || echo "")

  if [ "$TASK_STATUS" == "RUNNING" ] && [ "$AGENT_STATUS" == "RUNNING" ]; then
    echo "    OK: task RUNNING, agente ECS Exec RUNNING."
    break
  fi
  sleep 5
  if [ "$attempt" -eq 30 ]; then
    echo "[ERRORE] Timeout (150s) in attesa che il task/agente siano pronti."
    echo "         Stato task: $TASK_STATUS | Stato agente: $AGENT_STATUS"
    echo "         Puoi controllare manualmente con:"
    echo "           aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --region $REGION"
    exit 1
  fi
done

echo "==> [6/6] Apro la sessione interattiva (identica al prompt che vedi in locale)..."
echo "    Digita 1-7 o 'all' come faresti normalmente. Per uscire dalla sessione: exit / Ctrl+D."
echo ""

aws ecs execute-command \
  --cluster "$CLUSTER_NAME" \
  --task "$TASK_ARN" \
  --container "$CONTAINER_NAME" \
  --interactive \
  --command "python -m src.testing.engine" \
  --region "$REGION"

echo ""
echo "===================================================================="
echo " Sessione terminata."
echo " Se hai aggiunto l'upload su S3 in engine.py, il report finale è in:"
echo "   s3://$BUCKET_NAME/test_reports/..."
echo "===================================================================="
read -p "Fermare ora il task del test-engine? (Consigliato, evita costi Fargate inutili) [Y/n] " STOP_ANSWER
STOP_ANSWER="${STOP_ANSWER:-Y}"
if [[ "$STOP_ANSWER" =~ ^[Yy]$ ]]; then
  aws ecs stop-task --cluster "$CLUSTER_NAME" --task "$TASK_ARN" --region "$REGION" > /dev/null
  echo "Task fermato."
else
  echo "Task lasciato in esecuzione. Per fermarlo dopo:"
  echo "  aws ecs stop-task --cluster $CLUSTER_NAME --task $TASK_ARN --region $REGION"
fi