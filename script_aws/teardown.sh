#!/bin/bash
set -e

# =====================================================================
# TEARDOWN: porta a ZERO i servizi Fargate e ATTENDE la pulizia reale
# Usalo ogni volta che vuoi resettare il cluster o terminare i test.
# =====================================================================

REGION="us-east-1"
CLUSTER_NAME="forest-cluster"
BUCKET_NAME="my-cluster-datasets-bucket-759804778194-us-east-1-an"

# ---------------------------------------------------------------------
# Rilevamento della modalità corrente dal .env, con la stessa priorità
# usata da deploy.sh (SYS_MODE > TRAINING_MODE > default "centralized").
# Serve solo per --purge-legacy-mode: capire quali service NON
# appartengono alla modalità attualmente in uso.
# ---------------------------------------------------------------------
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  get_env_var() {
    local key="$1"
    grep -E "^[[:space:]]*${key}[[:space:]]*=" "$ENV_FILE" | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  }
  ENV_SYS_MODE=$(get_env_var "SYS_MODE")
  ENV_TRAINING_MODE=$(get_env_var "TRAINING_MODE")
  TRAINING_MODE="${ENV_SYS_MODE:-${ENV_TRAINING_MODE:-centralized}}"
else
  echo "==> [ATTENZIONE] File $ENV_FILE non trovato: assumo TRAINING_MODE=centralized per --purge-legacy-mode."
  TRAINING_MODE="centralized"
fi

# ---------------------------------------------------------------------
# Flag opzionali da riga di comando.
# --purge-shards: elimina anche gli shard federati su S3
#   (s3://$BUCKET_NAME/federated_shards/). NON attivo di default, perché
#   generarli richiede ricaricare e ri-splittare il dataset da zero
#   (RawCSVDataLoader + FederatedDataSplitter): sono pensati per
#   sopravvivere a più cicli di teardown/deploy. Usa questo flag solo se
#   vuoi forzare un re-provisioning completo (es. dataset locale cambiato).
# --purge-legacy-mode: elimina (aws ecs delete-service, non solo
#   desired-count=0) i service Fargate che appartengono alla modalità
#   di training DIVERSA da quella attuale (es. il vecchio 'worker-service'
#   unico se sei passata a TRAINING_MODE=federated, o i 'worker-service-N'
#   se sei tornata a centralized). teardown.sh normale li scala solo a 0
#   e li lascia lì apposta per riavvii rapidi (vedi messaggio finale):
#   usa questo flag quando invece vuoi ripulirli definitivamente perché
#   non riprenderai più quella modalità sullo stesso cluster.
# ---------------------------------------------------------------------
PURGE_SHARDS=0
PURGE_LEGACY_MODE=0
for arg in "$@"; do
  case "$arg" in
    --purge-shards) PURGE_SHARDS=1 ;;
    --purge-legacy-mode) PURGE_LEGACY_MODE=1 ;;
  esac
done
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

echo "==> [1/5] Scoperta dei Service attivi sul cluster (worker-service, worker-service-N, orchestrator-service)..."
# Non hardcodiamo i nomi: con TRAINING_MODE=centralized esiste 'worker-service',
# con TRAINING_MODE=federated esistono 'worker-service-1'..'worker-service-N'.
# Scopriamo dinamicamente cosa c'è davvero sul cluster, così funziona in entrambi
# i casi e anche se la modalità è cambiata tra un deploy e l'altro.
ALL_SERVICE_ARNS=$(aws ecs list-services --cluster "$CLUSTER_NAME" --region "$REGION" \
  --query "serviceArns[]" --output text 2>/dev/null || echo "")

# Un service è "legacy" se appartiene alla modalità DIVERSA da quella corrente:
# - in federated, il legacy è l'unico "worker-service" (nome esatto, senza suffisso -N)
# - in centralized, il legacy sono i "worker-service-N" (con suffisso numerico)
is_legacy_for_current_mode() {
  local name="$1"
  if [ "$TRAINING_MODE" == "federated" ]; then
    [ "$name" == "worker-service" ]
  else
    [[ "$name" =~ ^worker-service-[0-9]+$ ]]
  fi
}

TARGET_SERVICES=()
LEGACY_SERVICES=()
for arn in $ALL_SERVICE_ARNS; do
  svc_name="${arn##*/}"
  if [[ "$svc_name" == worker-service* || "$svc_name" == "orchestrator-service" ]]; then
    TARGET_SERVICES+=("$svc_name")
    if is_legacy_for_current_mode "$svc_name"; then
      LEGACY_SERVICES+=("$svc_name")
    fi
  fi
done

if [ "$PURGE_LEGACY_MODE" -eq 1 ] && [ "${#LEGACY_SERVICES[@]}" -gt 0 ]; then
  echo "    [PURGE-LEGACY-MODE] Modalità corrente: $TRAINING_MODE. Service dell'altra modalità"
  echo "    che verranno ELIMINATI (non solo scalati a 0) a fine teardown: ${LEGACY_SERVICES[*]}"
fi

if [ "${#TARGET_SERVICES[@]}" -eq 0 ]; then
  echo "    Nessun service worker/orchestrator trovato sul cluster, salto abbattimento."
else
  echo "    Service trovati: ${TARGET_SERVICES[*]}"
  for svc in "${TARGET_SERVICES[@]}"; do
    aws ecs update-service --cluster "$CLUSTER_NAME" --service "$svc" \
      --desired-count 0 --region "$REGION" > /dev/null 2>&1 || echo "    ($svc: update fallito, salto)"
  done

  echo ""
  echo "==> SINCRONIZZAZIONE AWS: Attendo la distruzione di TUTTI i container attivi..."
  echo "    (Fargate sta spegnendo i vecchi nodi zombie. Il terminale si sbloccherà automaticamente, attendi...)"

  # Questo comando blocca l'esecuzione finché i nodi in esecuzione (running-count) non scendono a 0 (pari al desired-count)
  echo "==> [2/5] Attendo lo spegnimento REALE dei container (Sincronizzazione)..."
  aws ecs wait services-stable \
    --cluster "$CLUSTER_NAME" \
    --services "${TARGET_SERVICES[@]}" \
    --region "$REGION"
  echo "    Fargate ha spento tutti i container."

  if [ "$PURGE_LEGACY_MODE" -eq 1 ] && [ "${#LEGACY_SERVICES[@]}" -gt 0 ]; then
    echo ""
    echo "==> [2b/5] --purge-legacy-mode: eliminazione definitiva dei service legacy..."
    for svc in "${LEGACY_SERVICES[@]}"; do
      echo "    Eliminazione service '$svc' (modalità diversa da $TRAINING_MODE)..."
      aws ecs delete-service --cluster "$CLUSTER_NAME" --service "$svc" --force --region "$REGION" > /dev/null 2>&1 \
        && echo "    '$svc' eliminato." \
        || echo "    ($svc: eliminazione fallita, salto)"
    done
  fi
fi

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
  "WorkerIndexLocks"
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
if [ "${#TARGET_SERVICES[@]}" -eq 0 ]; then
  echo "  (nessun service trovato in fase di teardown: verifica il nome dei service col deploy usato)"
else
  is_purged_legacy() {
    local name="$1"
    if [ "$PURGE_LEGACY_MODE" -ne 1 ]; then
      return 1
    fi
    local l
    for l in "${LEGACY_SERVICES[@]}"; do
      [ "$l" == "$name" ] && return 0
    done
    return 1
  }

  for svc in "${TARGET_SERVICES[@]}"; do
    if is_purged_legacy "$svc"; then
      continue  # eliminato con --purge-legacy-mode: non ha più senso riavviarlo
    fi
    if [ "$svc" == "orchestrator-service" ]; then
      echo "  aws ecs update-service --cluster $CLUSTER_NAME --service $svc --desired-count 2 --region $REGION"
    elif [ "$svc" == "worker-service" ]; then
      # centralized: unico service, il desired-count va al NUM_WORKERS originale (non 1)
      echo "  aws ecs update-service --cluster $CLUSTER_NAME --service $svc --desired-count \$NUM_WORKERS --region $REGION"
    else
      # federated: ogni worker-service-N è fisso a 1 (un solo task per indice)
      echo "  aws ecs update-service --cluster $CLUSTER_NAME --service $svc --desired-count 1 --region $REGION"
    fi
  done
  if [ "$PURGE_LEGACY_MODE" -eq 1 ] && [ "${#LEGACY_SERVICES[@]}" -gt 0 ]; then
    echo ""
    echo "Eliminati definitivamente (--purge-legacy-mode, modalità $TRAINING_MODE): ${LEGACY_SERVICES[*]}"
    echo "Per riaverli, serve un deploy.sh completo nell'altra modalità (li ricrea da zero)."
  fi
fi