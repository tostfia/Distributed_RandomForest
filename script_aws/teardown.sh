#!/bin/bash
set -e

# =====================================================================
# TEARDOWN: porta a ZERO i servizi Fargate (senza eliminare cluster/immagini)
# Usalo ogni volta che finisci una sessione di lavoro, per non consumare
# crediti mentre non stai testando attivamente.
# =====================================================================

REGION="us-east-1"
CLUSTER_NAME="forest-cluster"

echo "==> Azzero il Service worker-service..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service worker-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (worker-service non trovato, salto)"

echo "==> Azzero il Service orchestrator-service..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service orchestrator-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (orchestrator-service non trovato, salto)"

echo ""
echo "Fatto. I servizi restano configurati (task definition, cluster, ECR)"
echo "ma desired-count=0 significa nessun task Fargate attivo -> nessun costo di compute."
echo ""
echo "Per far ripartire tutto senza rifare il deploy da capo:"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service worker-service --desired-count 2 --region $REGION"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service orchestrator-service --desired-count 2 --region $REGION"