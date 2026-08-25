#!/usr/bin/env bash
# =====================================================================
# upload_dataset.sh
#
# Carica il dataset locale su S3 nella cartella real/ del bucket creato
# da Terraform, con retry automatico per gestire reti lente/instabili.
#
# Uso:
#   ./upload_dataset.sh                     # usa dataset_cache/ di default
#   ./upload_dataset.sh /percorso/dataset    # percorso locale custom
# =====================================================================
set -uo pipefail

# --- Configurazione ---
TERRAFORM_DIR="${TERRAFORM_DIR:-$(dirname "$0")/terraform}"
LOCAL_DATASET_DIR="${1:-$(dirname "$0")/dataset_cache}"
S3_PREFIX="real/"
MAX_ATTEMPTS=10
RETRY_DELAY_SECONDS=15

echo "==> Recupero nome bucket da Terraform..."
if [ ! -d "$TERRAFORM_DIR" ]; then
    echo "ERRORE: directory Terraform non trovata in $TERRAFORM_DIR" >&2
    echo "Imposta TERRAFORM_DIR se il percorso è diverso." >&2
    exit 1
fi

BUCKET=$(cd "$TERRAFORM_DIR" && terraform output -raw datasets_bucket_name 2>/dev/null)
if [ -z "$BUCKET" ]; then
    echo "ERRORE: impossibile leggere l'output 'datasets_bucket_name' da Terraform." >&2
    echo "Verifica di aver già fatto 'terraform apply' con successo." >&2
    exit 1
fi
echo "==> Bucket: $BUCKET"

if [ ! -d "$LOCAL_DATASET_DIR" ]; then
    echo "ERRORE: cartella dataset locale non trovata: $LOCAL_DATASET_DIR" >&2
    exit 1
fi

echo "==> Sorgente locale: $LOCAL_DATASET_DIR"
echo "==> Destinazione:    s3://$BUCKET/$S3_PREFIX"

# --- Tuning CLI per reti lente ---
aws configure set default.s3.max_concurrent_requests 10
aws configure set default.s3.multipart_chunksize 32MB

# --- Upload con retry ---
attempt=1
success=0
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo ""
    echo "=== Tentativo $attempt/$MAX_ATTEMPTS ==="
    if aws s3 sync "$LOCAL_DATASET_DIR" "s3://$BUCKET/$S3_PREFIX" \
        --exclude "*.tmp" --exclude "*.log"; then
        success=1
        echo "=== Upload completato al tentativo $attempt ==="
        break
    fi
    echo "=== Tentativo $attempt fallito, ritento tra ${RETRY_DELAY_SECONDS}s ==="
    sleep "$RETRY_DELAY_SECONDS"
    attempt=$((attempt + 1))
done

if [ "$success" -ne 1 ]; then
    echo "ERRORE: upload non riuscito dopo $MAX_ATTEMPTS tentativi." >&2
    echo "Rilancia lo script più tardi: 'sync' riprenderà solo i file mancanti." >&2
    exit 1
fi

# --- Verifica finale ---
echo ""
echo "==> Verifica contenuto caricato:"
aws s3 ls "s3://$BUCKET/$S3_PREFIX" --recursive --summarize | tail -n 5

echo ""
echo "==> Fatto. Dataset disponibile su s3://$BUCKET/$S3_PREFIX"