#!/bin/bash
set -e

# =====================================================================
# RUN TEST ENGINE su EC2 on-demand (invece che come task ECS Fargate).
#
# Perché: la SCP del Learner Lab nega 'ecs:RegisterTaskDefinition' con
# memoria > 8192 MiB (sia FARGATE che EC2-backed), ma 'ec2:RunInstances'
# su un tipo whitelisted (r5.large, 16 GiB) non è soggetto a questa
# restrizione. Il test-engine crea un CentralizedOrchestrator IN-PROCESS
# (vedi engine.py) e con carichi di scalabilità pesanti (molti worker,
# n_estimators alto) può superare 8 GiB — qui gli diamo 16 GiB.
#
# L'istanza è USA E GETTA: lo user-data avvia il container, attende che
# finisca, poi la termina da sola (shutdown -h now dopo il container).
# Nessuna sessione interattiva richiesta: stessa logica di
# run_test_engine_ecs.sh (SCENARIO passato come variabile d'ambiente).
#
# Prerequisiti: uguali a run_test_engine_ecs.sh (.env con ENV_MODE=aws,
# credenziali AWS Academy correnti, worker-service RUNNING sul cluster).
# In più: log group CloudWatch creato a mano una tantum:
#   aws logs create-log-group --log-group-name "/ec2/rf-test-engine" --region us-east-1
#
# Uso:
#   ./run_test_engine.sh <scenario>      # es. ./run_test_engine.sh 2
#   ./run_test_engine.sh                 # chiede lo scenario a terminale
# =====================================================================

if [ ! -f .env ]; then
  echo "[ERRORE] File .env non trovato nella directory corrente ($(pwd))."
  exit 1
fi

get_env_var() {
  local key="$1"
  grep -E "^[[:space:]]*${key}[[:space:]]*=" .env | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}

ENV_ENV_MODE=$(get_env_var "ENV_MODE")
ENV_TRAINING_MODE=$(get_env_var "TRAINING_MODE")
ENV_NUM_WORKERS=$(get_env_var "NUM_WORKERS")
ENV_REGION=$(get_env_var "AWS_DEFAULT_REGION")
ENV_BUCKET_NAME=$(get_env_var "DATASETS_BUCKET_NAME")
ENV_RPC_SYNC_TIMEOUT=$(get_env_var "RPC_SYNC_TIMEOUT_SECONDS")
ENV_RPC_INFERENCE_SYNC_TIMEOUT=$(get_env_var "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS")
RPC_SYNC_TIMEOUT_SECONDS="${ENV_RPC_SYNC_TIMEOUT:-1800}"
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS="${ENV_RPC_INFERENCE_SYNC_TIMEOUT:-900}"

if [ "$ENV_ENV_MODE" != "aws" ]; then
  echo "[ERRORE] ENV_MODE nel .env è '$ENV_ENV_MODE', non 'aws'."
  exit 1
fi

TRAINING_MODE="${ENV_TRAINING_MODE:-centralized}"
REGION="${ENV_REGION:-us-east-1}"
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"
if [ -z "$ENV_BUCKET_NAME" ]; then
  echo "[ERRORE] DATASETS_BUCKET_NAME non impostato nel .env."
  exit 1
fi
BUCKET_NAME="$ENV_BUCKET_NAME"
CLUSTER_NAME="forest-cluster"
REPO_NAME="rf-distributed"
SG_NAME="rf-distributed-sg"
INSTANCE_TYPE="r5.large"
# r5.large non è offerto in TUTTE le AZ di questo account (verificato
# empiricamente: 'RunInstances' fallisce con 'Unsupported' in us-east-1e,
# messaggio d'errore che elenca esplicitamente le AZ supportate). Stessa
# lista usata nel filtro Terraform (orchestrator_ec2.tf).
SUPPORTED_AZS="us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1f"
# Verificata valida per us-east-1 il 5/9/2026. Le AMI Amazon Linux vengono
# ruotate periodicamente: se questo ID inizia a dare 'InvalidAMIID.NotFound',
# rilanciare:
#   aws ec2 describe-images --owners amazon \
#     --filters "Name=name,Values=al2023-ami-*-x86_64" "Name=state,Values=available" \
#     --query "sort_by(Images,&CreationDate)[-1].ImageId" --region us-east-1
AMI_ID="ami-0ac62d2d72afdce51"

if [[ "$TRAINING_MODE" != "centralized" && "$TRAINING_MODE" != "federated" ]]; then
  echo "ERRORE: TRAINING_MODE deve essere 'centralized' o 'federated', ricevuto: '$TRAINING_MODE'"
  exit 1
fi

VALID_SCENARIOS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "all")
SCENARIO_CHOICE="$1"

is_valid_scenario() {
  local candidate="$1"
  for v in "${VALID_SCENARIOS[@]}"; do
    [ "$v" == "$candidate" ] && return 0
  done
  return 1
}

if [ -z "$SCENARIO_CHOICE" ]; then
  echo "Seleziona lo scenario da eseguire:"
  echo "1. Performance e Metriche"
  echo "2. Scalabilità"
  echo "3. Simulazione di Rete"
  echo "4. Guasto improvviso del Worker (addestramento)"
  echo "5. Guasto improvviso del Worker (inferenza)"
  echo "6. Failover dell'Orchestratore (addestramento)"
  echo "7. Failover dell'Orchestratore (inferenza)"
  echo "8. Elezione del Leader sotto Concorrenza (Safety)"
  echo "9. Genera Grafici"
  while true; do
    read -p "Scelta (1-9, o 'all' per eseguire tutti): " SCENARIO_CHOICE
    SCENARIO_CHOICE="$(echo "$SCENARIO_CHOICE" | tr '[:upper:]' '[:lower:]' | xargs)"
    is_valid_scenario "$SCENARIO_CHOICE" && break
    echo "[ERRORE] Opzione non valida. Riprova."
  done
else
  SCENARIO_CHOICE="$(echo "$SCENARIO_CHOICE" | tr '[:upper:]' '[:lower:]' | xargs)"
  if ! is_valid_scenario "$SCENARIO_CHOICE"; then
    echo "[ERRORE] Scenario '$SCENARIO_CHOICE' non valido. Valori ammessi: ${VALID_SCENARIOS[*]}"
    exit 1
  fi
fi

INSTANCE_ID=""

cleanup_on_failure() {
  local exit_code=$?
  if [ -n "$INSTANCE_ID" ] && [ "$exit_code" -ne 0 ]; then
    echo ""
    echo "[CLEANUP] Errore rilevato: termino l'istanza EC2 ($INSTANCE_ID) per non lasciarla accesa..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" > /dev/null 2>&1 \
      && echo "[CLEANUP] Istanza terminata." \
      || echo "[CLEANUP] (istanza già terminata o non trovata: ok)"
  fi
  exit $exit_code
}
trap cleanup_on_failure EXIT INT TERM HUP

echo "===================================================================="
echo " RUN TEST ENGINE (EC2 on-demand, $INSTANCE_TYPE)  -  ($REGION)"
echo "===================================================================="
echo " TRAINING_MODE : $TRAINING_MODE"
echo " NUM_WORKERS   : $NUM_WORKERS"
echo " SCENARIO      : $SCENARIO_CHOICE"
echo "--------------------------------------------------------------------"

echo "==> [1/4] Verifica credenziali AWS (Learner Lab, scadono ogni ~4h)..."
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute. Aggiornale ed esportale nella shell, poi riprova."
  exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo "    OK, credenziali valide."

echo "==> [2/4] Verifica che worker-service sia stabile e l'orchestrator (EC2) sia pronto..."
if ! aws ecs describe-services --cluster "$CLUSTER_NAME" --services worker-service --region "$REGION" \
     --query "services[0].status" --output text 2>/dev/null | grep -q ACTIVE; then
  echo "[ERRORE] worker-service non trovato/attivo sul cluster '$CLUSTER_NAME'. Hai lanciato 'terraform apply'?"
  exit 1
fi
aws ecs wait services-stable --cluster "$CLUSTER_NAME" --services worker-service --region "$REGION"

ORCH_RUNNING=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Project,Values=rf-distributed" "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null || echo "")
ORCH_COUNT=$(echo "$ORCH_RUNNING" | wc -w)
if [ "$ORCH_COUNT" -eq 0 ]; then
  echo "    [ATTENZIONE] Nessuna istanza EC2 dell'orchestrator RUNNING. Gli scenari 6/7 (failover)"
  echo "                 falliranno; gli altri scenari non ne hanno bisogno (orchestratore in-process)."
else
  echo "    Istanze EC2 orchestrator RUNNING: $ORCH_COUNT ($ORCH_RUNNING)"
fi
echo "    OK, worker pronti."

echo "==> [3/4] Recupero VPC/subnet/Security Group (stessi usati da Terraform)..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")
# Filtro esplicito sulle AZ compatibili con r5.large (vedi SUPPORTED_AZS
# sopra): senza questo, 'Subnets[0]' può capitare in us-east-1e, dove
# r5.large non è disponibile (bug reale già incontrato una volta con
# questo stesso script, e risolto in Terraform con lo stesso filtro).
SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
    "Name=availability-zone,Values=${SUPPORTED_AZS}" \
  --query "Subnets[0].SubnetId" --output text --region "$REGION")
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION")
if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
  echo "[ERRORE] Security Group '$SG_NAME' non trovato: hai lanciato 'terraform apply'?"
  exit 1
fi
if [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" == "None" ]; then
  echo "[ERRORE] Nessuna subnet pubblica trovata in una AZ compatibile con $INSTANCE_TYPE ($SUPPORTED_AZS)."
  exit 1
fi
echo "    VPC: $VPC_ID | Subnet: $SUBNET_ID | SG: $SG_ID"

echo "==> [4/4] Avvio dell'istanza EC2 ($INSTANCE_TYPE, 16 GiB)..."
USER_DATA=$(cat <<EOF
#!/bin/bash
set -e
yum install -y docker
systemctl enable docker
systemctl start docker

aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${ECR_REGISTRY}

docker run --rm \
  --name test-engine \
  --log-driver awslogs \
  --log-opt awslogs-region=${REGION} \
  --log-opt awslogs-group=/ec2/rf-test-engine \
  --log-opt awslogs-create-group=false \
  --log-opt awslogs-stream=test-engine-\$(date +%s) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e NUM_WORKERS=${NUM_WORKERS} \
  -e ENV_MODE=aws \
  -e TRAINING_MODE=${TRAINING_MODE} \
  -e EC2_ID=EC2TestEngine \
  -e RUNNING_IN_DOCKER=true \
  -e AWS_DEFAULT_REGION=${REGION} \
  -e DATASETS_BUCKET_NAME=${BUCKET_NAME} \
  -e RPC_SYNC_TIMEOUT_SECONDS=${RPC_SYNC_TIMEOUT_SECONDS}s \
  -e RPC_INFERENCE_SYNC_TIMEOUT_SECONDS=${RPC_INFERENCE_SYNC_TIMEOUT_SECONDS}s \
  -e SCENARIO=${SCENARIO_CHOICE} \
  ${ECR_REGISTRY}/${REPO_NAME}:latest \
  sh -c "timeout 7200 python -m src.testing.engine"

# Fine del container: l'istanza usa-e-getta si ferma da sola, così non
# resta accesa (e fatturata) dopo che il test è concluso.
shutdown -h now
EOF
)

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name=LabInstanceProfile \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rf-test-engine-ec2},{Key=Project,Value=rf-distributed}]" \
  --region "$REGION" \
  --query "Instances[0].InstanceId" --output text)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "None" ]; then
  echo "[ERRORE] run-instances non ha restituito un Instance ID valido."
  exit 1
fi
echo "    Istanza avviata: $INSTANCE_ID"

echo "    Attendo che l'istanza sia RUNNING..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "    OK: istanza RUNNING. Il container si avvia dopo qualche secondo (Docker da installare)."

trap - EXIT INT TERM HUP

echo ""
echo "===================================================================="
echo " Istanza avviata con successo. Il test prosegue in background;"
echo " l'istanza si TERMINA DA SOLA a fine test (shutdown automatico)."
echo "===================================================================="
echo " Per seguire i log in tempo reale (il gruppo va creato una tantum,"
echo " vedi commento in testa allo script):"
echo "   aws logs tail /ec2/rf-test-engine --follow --region $REGION"
echo ""
echo " Per controllare se l'istanza ha terminato:"
echo "   aws ec2 describe-instances --instance-ids $INSTANCE_ID \\"
echo "     --query \"Reservations[0].Instances[0].State.Name\" --region $REGION"
echo ""
echo " Il report finale viene caricato in:"
echo "   s3://$BUCKET_NAME/test_reports/aws/"
echo ""
echo " Per fermarla manualmente prima che finisca:"
echo "   aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo "===================================================================="