#!/bin/bash
set -e

# =====================================================================
# TEARDOWN: porta a ZERO i servizi Fargate e ATTENDE la pulizia reale
# Usalo ogni volta che vuoi resettare il cluster o terminare i test.
# =====================================================================

if ! command -v jq > /dev/null 2>&1; then
  echo "ERRORE: 'jq' non è installato. Necessario per la pulizia delle tabelle DynamoDB."
  echo "Installa con: pip install jq"
  exit 1
fi

REGION="us-east-1"
CLUSTER_NAME="forest-cluster"
BUCKET_NAME="my-cluster-datasets-bucket-759804778194-us-east-1-an"

# ---------------------------------------------------------------------
# Svuota TUTTI gli item di una tabella DynamoDB, senza cancellare la
# tabella stessa (schema, throughput, ecc. restano intatti).
# Richiede 'jq' installato sulla macchina che lancia lo script.
# ---------------------------------------------------------------------
purge_dynamodb_table() {
  local table="$1"

  if ! aws dynamodb describe-table --table-name "$table" --region "$REGION" > /dev/null 2>&1; then
    echo "    (Tabella '$table' non trovata, salto)"
    return
  fi

  local key_attrs
  key_attrs=$(aws dynamodb describe-table --table-name "$table" --region "$REGION" \
    --query "Table.KeySchema[].AttributeName" --output json | jq -r 'join(",")')

  local items
  items=$(aws dynamodb scan --table-name "$table" --region "$REGION" \
    --projection-expression "$key_attrs" --output json | jq -c '.Items[]')

  if [ -z "$items" ]; then
    echo "    Tabella '$table' già vuota."
    return
  fi

  local removed=0
  while IFS= read -r item; do
    aws dynamodb delete-item --table-name "$table" --region "$REGION" --key "$item" > /dev/null 2>&1
    removed=$((removed + 1))
  done <<< "$items"

  echo "    Tabella '$table' svuotata: $removed elementi rimossi."
}

echo "==> [1/5] Abbattimento nodi computazionali Fargate..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service worker-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (worker-service non trovato, salto)"

echo "==> Richiedo l'azzeramento del Service orchestrator-service..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service orchestrator-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (orchestrator-service non trovato, salto)"

echo ""
echo "==> SINCRONIZZAZIONE AWS: Attendo la distruzione di TUTTI i container attivi..."
echo "    (Fargate sta spegnendo i vecchi nodi zombie. Il terminale si sbloccherà automaticamente, attendi...)"

# Questo comando blocca l'esecuzione finché i nodi in esecuzione (running-count) non scendono a 0 (pari al desired-count)
echo "==> [2/5] Attendo lo spegnimento REALE dei container (Sincronizzazione)..."
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services worker-service orchestrator-service \
  --region "$REGION"
echo "    Fargate ha spento tutti i container."

echo "==> [3/5] Svuotamento stato applicativo su DynamoDB..."
echo "    Tabelle: workers_registry, orchestrators_registry, JobLocks, ModelStatus, OrchestratorLocks, WorkerTasks"
echo "    Nota: vengono rimossi solo gli ITEM, le tabelle restano intatte."

# Elenco confermato dalla console DynamoDB (6 tabelle totali usate dal sistema)
DYNAMO_TABLES=(
  "workers_registry"
  "orchestrators_registry"
  "JobLocks"
  "ModelStatus"
  "OrchestratorLocks"
  "WorkerTasks"
)

for t in "${DYNAMO_TABLES[@]}"; do
  purge_dynamodb_table "$t"
done

echo "==> [4/5] Svuotamento Code SQS FIFO..."
QUEUES=("centralized_queue.fifo" "federated_queue.fifo")
for q in "${QUEUES[@]}"; do
  URL=$(aws sqs get-queue-url --queue-name "$q" --query "QueueUrl" --output text --region "$REGION" 2>/dev/null || echo "None")
  if [ "$URL" != "None" ] && [ ! -z "$URL" ]; then
    # Il comando purge elimina istantaneamente tutti i messaggi nella coda
    aws sqs purge-queue --queue-url "$URL" --region "$REGION" 2>/dev/null || echo "    (Coda $q già vuota o purgatata di recente)"
    echo "    Coda $q svuotata con successo."
  fi
done

echo "==> [5/5] Pulizia file temporanei su S3..."
# Cancella SOLTANTO il contenuto della sottocartella temp/
aws s3 rm "s3://$BUCKET_NAME/temp/" --recursive --region "$REGION" > /dev/null 2>&1 || echo "    (Nessun file temporaneo da rimuovere)"
echo "    S3 ripulito: salvaguardati i dati in 'real/' e i modelli in 'models/'."

echo ""
echo "========================================================================"
echo " PULIZIA E RESUBMISSION REALE COMPLETATE"
echo "========================================================================"
echo "Fatto! AWS Fargate ha rimosso con successo ogni container dal cluster."
echo "Le tabelle DynamoDB (workers/orchestrators/job) sono state svuotate: il prossimo"
echo "deploy riparte da una tela pulita, senza job o registrazioni fantasma."
echo "I dataset in 'real/' e i modelli '.pkl' in 'models/' sono al sicuro."
echo "I servizi restano configurati (task definition, cluster, ECR)"
echo "ma desired-count=0 significa nessun task Fargate attivo -> nessun costo di compute."
echo ""
echo "Per far ripartire tutto senza rifare il deploy da capo:"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service worker-service --desired-count 2 --region $REGION"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service orchestrator-service --desired-count 2 --region $REGION"