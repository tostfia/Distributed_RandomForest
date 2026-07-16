#!/bin/bash
set -e

# =====================================================================
# TEARDOWN: porta a ZERO i servizi Fargate e ATTENDE la pulizia reale
# Usalo ogni volta che vuoi resettare il cluster o terminare i test.
# =====================================================================

REGION="us-east-1"
CLUSTER_NAME="forest-cluster"

echo "==> Richiedo l'azzeramento del Service worker-service..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service worker-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (worker-service non trovato, salto)"

echo "==> Richiedo l'azzeramento del Service orchestrator-service..."
aws ecs update-service --cluster "$CLUSTER_NAME" --service orchestrator-service \
  --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    (orchestrator-service non trovato, salto)"

echo ""
echo "==> SINCRONIZZAZIONE AWS: Attendo la distruzione di TUTTI i container attivi..."
echo "    (Fargate sta spegnendo i vecchi nodi zombie. Il terminale si sbloccherà automaticamente, attendi...)"

# Questo comando blocca l'esecuzione finché i nodi in esecuzione (running-count) non scendono a 0 (pari al desired-count)
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services worker-service orchestrator-service \
  --region "$REGION"

echo ""
echo "========================================================================"
echo " PULIZIA E RESUBMISSION REALE COMPLETATE"
echo "========================================================================"
echo "Fatto! AWS Fargate ha rimosso con successo ogni container dal cluster."
echo "I servizi restano configurati (task definition, cluster, ECR)"
echo "ma desired-count=0 significa nessun task Fargate attivo -> nessun costo di compute."
echo ""
echo "Per far ripartire tutto senza rifare il deploy da capo:"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service worker-service --desired-count 2 --region $REGION"
echo "  aws ecs update-service --cluster $CLUSTER_NAME --service orchestrator-service --desired-count 2 --region $REGION"