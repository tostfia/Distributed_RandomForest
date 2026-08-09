#!/bin/bash
set -e

# =====================================================================
# TEARDOWN: porta a ZERO i servizi Fargate e ATTENDE la pulizia reale
# Usalo ogni volta che vuoi resettare il cluster o terminare i test.
# =====================================================================

REGION="us-east-1"
CLUSTER_NAME="forest-cluster"
BUCKET_NAME="my-cluster-datasets-bucket-759804778194-us-east-1-an"

echo "==> [1/4] Abbattimento nodi computazionali Fargate..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service worker-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (worker-service non trovato, salto)"

echo "==> Richiedo l'azzeramento del Service orchestrator-service..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service orchestrator-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (orchestrator-service non trovato, salto)"

echo ""
echo "==> SINCRONIZZAZIONE AWS: Attendo la distruzione di TUTTI i container attivi..."
echo "    (Fargate sta spegnendo i vecchi nodi zombie. Il terminale si sbloccherà automaticamente, attendi...)"

# Questo comando blocca l'esecuzione finché i nodi in esecuzione (running-count) non scendono a 0 (pari al desired-count)
echo "==> [2/4] Attendo lo spegnimento REALE dei container (Sincronizzazione)..."
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services worker-service orchestrator-service \
  --region "$REGION"
echo "    Fargate ha spento tutti i container."

echo "==> [3/4] Svuotamento Code SQS FIFO..."
QUEUES=("centralized_queue.fifo" "federated_queue.fifo")
for q in "${QUEUES[@]}"; do
  URL=$(aws sqs get-queue-url --queue-name "$q" --query "QueueUrl" --output text --region "$REGION" 2>/dev/null || echo "None")
  if [ "$URL" != "None" ] && [ ! -z "$URL" ]; then
    # Il comando purge elimina istantaneamente tutti i messaggi nella coda
    aws sqs purge-queue --queue-url "$URL" --region "$REGION" 2>/dev/null || echo "    (Coda $q già vuota o purgatata di recente)"
    echo "    Coda $q svuotata con successo."
  fi
done

echo "==> [4/4] Pulizia file temporanei su S3..."
# Cancella SOLTANTO il contenuto della sottocartella temp/
aws s3 rm "s3://$BUCKET_NAME/temp/" --recursive --region "$REGION" > /dev/null 2>&1 || echo "    (Nessun file temporaneo da rimuovere)"
echo "    S3 ripulito: salvaguardati i dati in 'real/' e i modelli in 'models/'."

echo ""
echo "========================================================================"
echo " PULIZIA E RESUBMISSION REALE COMPLETATE"
echo "========================================================================"
echo "Fatto! AWS Fargate ha rimosso con successo ogni container dal cluster."
echo "I dataset in 'real/' e i modelli '.pkl' in 'models/' sono al sicuro."
echo "I servizi restano configurati (task definition, cluster, ECR)"
echo "ma desired-count=0 significa nessun task Fargate attivo -> nessun costo di compute."
echo ""
echo "Per far ripartire tutto senza rifare il deploy da capo:"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service worker-service --desired-count 2 --region $REGION"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service orchestrator-service --desired-count 2 --region $REGION"