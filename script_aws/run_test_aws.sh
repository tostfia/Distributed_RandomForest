#!/bin/bash
set -e

# =====================================================================
# RUN TEST AWS: esegue la stessa suite di test (TestEngine) di run_test.sh,
# ma contro l'infrastruttura AWS ECS/Fargate già deployata con deploy.sh
# (analogo, per lo scopo, a run_aws.sh che invece lancia il client reale).
#
# A differenza di run_test.sh (che usa docker-compose per lanciare worker
# locali + il container test-engine), qui il TestEngine gira DIRETTAMENTE
# sulla tua macchina (nessun container aggiuntivo, nessun costo Fargate
# extra solo per testare): parla con S3/DynamoDB/SQS reali via le
# credenziali AWS Academy nel .env, e con gli orchestratori/worker reali
# già in esecuzione su ECS.
#
# Prerequisiti:
#   - deploy.sh già eseguito con successo (cluster 'forest-cluster' attivo,
#     service 'orchestrator-service' e 'worker-service'/'worker-service-N'
#     con i task RUNNING)
#   - .env nella root con SYS_ENV=aws e le credenziali AWS Academy correnti
#     (scadono ogni ~4h: se il Learner Lab è stato riavviato, aggiornale
#     PRIMA di lanciare questo script)
#   - dipendenze Python del progetto installate nell'ambiente locale
#     (incluso boto3, requests) — lo stesso ambiente usato da run_aws.sh
#
# Uso:
#   ./run_test_aws.sh
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

if [ "$ENV_SYS_ENV" != "aws" ]; then
  echo "[ERRORE] SYS_ENV nel .env è '$ENV_SYS_ENV', non 'aws'."
  echo "         Aggiorna il .env se vuoi far girare i test contro l'infrastruttura AWS."
  exit 1
fi

TRAINING_MODE="${ENV_SYS_MODE:-${ENV_TRAINING_MODE:-centralized}}"
REGION="${ENV_REGION:-us-east-1}"
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"
CLUSTER_NAME="forest-cluster"

if [[ "$TRAINING_MODE" != "centralized" && "$TRAINING_MODE" != "federated" ]]; then
  echo "ERRORE: SYS_MODE/TRAINING_MODE deve essere 'centralized' o 'federated', ricevuto: '$TRAINING_MODE'"
  exit 1
fi

echo "===================================================================="
echo " RUN TEST AWS  -  TestEngine contro forest-cluster ($REGION)"
echo "===================================================================="
echo " TRAINING_MODE : $TRAINING_MODE"
echo " NUM_WORKERS   : $NUM_WORKERS"
echo "--------------------------------------------------------------------"

echo "==> [1/4] Verifica credenziali AWS (Learner Lab, scadono ogni ~4h)..."
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute."
  echo "         Aggiorna le AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN nel .env"
  echo "         (o in ~/.aws/credentials, a seconda di come deploy.sh le ha impostate) e riprova."
  exit 1
fi
echo "    OK, credenziali valide."

echo "==> [2/4] Verifica che i Service ECS (worker + orchestrator) siano stabili..."
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
  echo "[ERRORE] Nessun Service worker/orchestrator trovato sul cluster '$CLUSTER_NAME'."
  echo "         Hai lanciato deploy.sh (con desired-count > 0) prima di questo script?"
  exit 1
fi

echo "    Service trovati: ${TARGET_SERVICES[*]}"
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services "${TARGET_SERVICES[@]}" \
  --region "$REGION"
echo "    OK, infrastruttura pronta (tutti i Service sono stabili)."

echo "==> [3/4] Verifica presenza task RUNNING per orchestrator-service (leader+standby)..."
RUNNING_ORCH=$(aws ecs list-tasks --cluster "$CLUSTER_NAME" --service-name orchestrator-service \
  --desired-status RUNNING --query "length(taskArns)" --output text --region "$REGION" 2>/dev/null || echo "0")
if [ "$RUNNING_ORCH" -lt 2 ]; then
  echo "[ATTENZIONE] Solo $RUNNING_ORCH task RUNNING su orchestrator-service (attesi 2: leader+standby)."
  echo "             Gli scenari di failover dell'orchestratore (6 e 7) potrebbero non avere uno"
  echo "             standby reale da verificare. Procedo comunque."
else
  echo "    OK: $RUNNING_ORCH task RUNNING su orchestrator-service."
fi

echo "==> [4/4] Avvio del TestEngine in locale (modalità: $TRAINING_MODE, ambiente: aws)..."
echo "    (Il TestEngine NON avvia worker locali: RUNNING_IN_DOCKER=true segnala che i worker"
echo "     sono già in esecuzione altrove, esattamente come già succede per il caso docker-compose."
echo "     Qui significa: sono i task reali di worker-service su Fargate.)"
echo ""

ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"

export RUNNING_IN_DOCKER=true
export ENV_MODE=aws
export SYS_ENV=aws
export TRAINING_MODE="$TRAINING_MODE"
export SYS_MODE="$TRAINING_MODE"
export NUM_WORKERS="$NUM_WORKERS"
export AWS_DEFAULT_REGION="$REGION"
export ECS_CLUSTER_NAME="$CLUSTER_NAME"

python -m src.testing.engine

echo ""
echo "===================================================================="
echo " TEST AWS COMPLETATI. Report in ./test_reports/docker/ "
echo " (stesso output_dir usato quando RUNNING_IN_DOCKER=true, vedi engine.py)"
echo "===================================================================="