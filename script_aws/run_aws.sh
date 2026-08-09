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

# Legge SYS_ENV/SYS_MODE dal .env solo per la verifica di coerenza qui sotto,
# SENZA esportarli: sarà config.py (via load_dotenv) a farlo al posto nostro,
# leggendo esattamente questi valori senza rischio di override dalla shell.
ENV_SYS_ENV=$(grep -E "^SYS_ENV=" .env | cut -d= -f2 | tr -d ' "')
ENV_SYS_MODE=$(grep -E "^SYS_MODE=" .env | cut -d= -f2 | tr -d ' "')

if [ "$ENV_SYS_ENV" != "aws" ]; then
  echo "[ERRORE] SYS_ENV nel .env è '$ENV_SYS_ENV', non 'aws'."
  echo "         Aggiorna il .env se vuoi puntare all'infrastruttura AWS."
  exit 1
fi

REGION=$(grep -E "^AWS_DEFAULT_REGION=" .env | cut -d= -f2 | tr -d ' "')
REGION="${REGION:-us-east-1}"

echo "==> [1/2] Verifica credenziali AWS (Learner Lab, scadono ogni ~4h)..."
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute."
  echo "         Aggiorna le AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN nel .env"
  echo "         con quelle correnti del Learner Lab e riprova."
  exit 1
fi
echo "    OK, credenziali valide."

echo "==> [2/2] Avvio Client (modalità dal .env: SYS_MODE=$ENV_SYS_MODE)..."
echo "    (Il client parla con l'infrastruttura solo via SQS/DynamoDB:"
echo "     nessun bisogno di conoscere IP o porte di orchestratori/worker su Fargate.)"
echo ""

ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"

python -m src.client.main