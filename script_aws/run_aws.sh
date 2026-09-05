#!/bin/bash
set -e

# =====================================================================
# RUN AWS CLIENT: avvia src.client.main contro l'infrastruttura AWS
# già deployata con deploy.sh (Fargate + SQS + DynamoDB + S3).
#
#
# Uso:
#   ./run_aws_client.sh
# =====================================================================

if [ ! -f .env ]; then
  echo "[ERRORE] File .env non trovato nella directory corrente ($(pwd))."
  exit 1
fi

# Legge ENV_MODE/TRAINING_MODE dal .env solo per la verifica di coerenza qui sotto,
# SENZA esportarli: sarà config.py (via load_dotenv) a farlo al posto nostro,
# leggendo esattamente questi valori senza rischio di override dalla shell.
# tr -d rimuove anche gli apici singoli oltre a quelli doppi, per coerenza
# con get_env_var() in deploy.sh (che gestisce entrambi i tipi di quoting).
ENV_ENV_MODE=$(grep -E "^ENV_MODE=" .env | cut -d= -f2 | tr -d " \"'")
ENV_TRAINING_MODE=$(grep -E "^TRAINING_MODE=" .env | cut -d= -f2 | tr -d " \"'")

if [ "$ENV_ENV_MODE" != "aws" ]; then
  echo "[ERRORE] ENV_MODE nel .env è '$ENV_ENV_MODE', non 'aws'."
  echo "         Aggiorna il .env se vuoi puntare all'infrastruttura AWS."
  exit 1
fi

REGION=$(grep -E "^AWS_DEFAULT_REGION=" .env | cut -d= -f2 | tr -d " \"'")
REGION="${REGION:-us-east-1}"
CLUSTER_NAME="forest-cluster"

echo "==> [1/3] Verifica credenziali AWS (Learner Lab, scadono ogni ~4h)..."
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute."
  echo "         Aggiorna le AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN nel .env"
  echo "         con quelle correnti del Learner Lab e riprova."
  exit 1
fi
echo "    OK, credenziali valide."

echo "==> [2/3] Attendo che worker-service sia stabile e verifico che l'orchestrator (EC2) sia pronto..."
echo "    (deploy asincrono: create/update-service ritorna subito, non quando i"
echo "     task sono RUNNING e i worker si sono registrati. Aspettiamo qui per evitare"
echo "     di sottomettere il job mentre l'infrastruttura sta ancora avviandosi.)"

ALL_SERVICE_ARNS=$(aws ecs list-services --cluster "$CLUSTER_NAME" --region "$REGION" \
  --query "serviceArns[]" --output text 2>/dev/null || echo "")

TARGET_SERVICES=()
for arn in $ALL_SERVICE_ARNS; do
  svc_name="${arn##*/}"
  if [[ "$svc_name" == worker-service* ]]; then
    TARGET_SERVICES+=("$svc_name")
  fi
done

if [ "${#TARGET_SERVICES[@]}" -eq 0 ]; then
  echo "[ERRORE] Nessun worker-service trovato sul cluster '$CLUSTER_NAME'."
  echo "         Hai lanciato 'terraform apply' prima di questo script?"
  exit 1
fi

echo "    Service worker trovati: ${TARGET_SERVICES[*]}"
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services "${TARGET_SERVICES[@]}" \
  --region "$REGION"

# L'orchestrator non è più un Service ECS (vedi orchestrator_ec2.tf): senza
# questo controllo, un job inviato qui sotto resterebbe silenziosamente in
# coda SQS finché nessuna istanza EC2 dell'orchestrator lo reclama — a
# differenza del vecchio 'services-stable' su ECS, qui il controllo è
# BLOCCANTE (exit 1), perché sottomettere un job senza orchestratore pronto
# non è un'attesa normale, è un errore d'uso.
ORCH_RUNNING=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Project,Values=rf-distributed" "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null || echo "")
ORCH_COUNT=$(echo "$ORCH_RUNNING" | wc -w)
if [ "$ORCH_COUNT" -eq 0 ]; then
  echo "[ERRORE] Nessuna istanza EC2 dell'orchestrator RUNNING: il job resterebbe in coda"
  echo "         SQS senza che nessuno lo reclami. Verifica con 'terraform apply' o"
  echo "         'aws ec2 describe-instances --filters Name=tag:Project,Values=rf-distributed --region $REGION'."
  exit 1
fi
echo "    Istanze EC2 orchestrator RUNNING: $ORCH_COUNT ($ORCH_RUNNING)"
echo "    OK, infrastruttura pronta."

echo "==> [3/3] Avvio Client (modalità dal .env: TRAINING_MODE=$ENV_TRAINING_MODE)..."
echo "    (Il client parla con l'infrastruttura solo via SQS/DynamoDB:"
echo "     nessun bisogno di conoscere IP o porte di orchestratori/worker su Fargate.)"
echo ""

ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"

python -m src.client.main