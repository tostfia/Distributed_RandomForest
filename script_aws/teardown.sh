#!/bin/bash
set -e

# =====================================================================
# TEARDOWN: porta a ZERO i servizi Fargate e ATTENDE la pulizia reale
# Usalo ogni volta che vuoi resettare il cluster o terminare i test.
# =====================================================================

REGION="us-east-1"
CLUSTER_NAME="forest-cluster"
DATASETS_BUCKET_NAME="rf-distributed-datasets-378857401407-us-east-1"

# ---------------------------------------------------------------------
# Rilevamento della modalità corrente dal .env, con la stessa priorità
# usata da Terraform (TRAINING_MODE > default "centralized").
# Serve solo per --purge-legacy-mode: capire quali service NON
# appartengono alla modalità attualmente in uso.
# ---------------------------------------------------------------------
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  get_env_var() {
    local key="$1"
    grep -E "^[[:space:]]*${key}[[:space:]]*=" "$ENV_FILE" | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  }
  ENV_TRAINING_MODE=$(get_env_var "TRAINING_MODE")
  TRAINING_MODE="${ENV_TRAINING_MODE:-centralized}"
else
  echo "==> [ATTENZIONE] File $ENV_FILE non trovato: assumo TRAINING_MODE=centralized per --purge-legacy-mode."
  TRAINING_MODE="centralized"
fi

echo "==> [0/6] Arresto di eventuali task one-off del test-engine ancora attivi..."
STRAY_ENGINE_TASKS=$(aws ecs list-tasks --cluster "$CLUSTER_NAME" \
  --family "rf-test-engine-task" --desired-status RUNNING \
  --query "taskArns[]" --output text --region "$REGION" 2>/dev/null || echo "")

if [ -n "$STRAY_ENGINE_TASKS" ]; then
  for task_arn in $STRAY_ENGINE_TASKS; do
    echo "    Trovato task-engine orfano ancora RUNNING: $task_arn"
    aws ecs stop-task --cluster "$CLUSTER_NAME" --task "$task_arn" --region "$REGION" > /dev/null 2>&1 \
      && echo "    Fermato." \
      || echo "    (stop fallito, verifica manualmente)"
  done
else
  echo "    Nessun task-engine orfano trovato."
fi

# ---------------------------------------------------------------------
# Flag opzionali da riga di comando.
# --purge-shards: elimina anche gli shard federati su S3
#   (s3://$DATASETS_BUCKET_NAME/federated_shards/). NON attivo di default, perché
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
# --purge-models: elimina anche i modelli salvati su S3
#   (s3://$DATASETS_BUCKET_NAME/saved_models/). NON attivo di default: sono
#   l'output vero e proprio dei training e per scelta vanno cancellati
#   solo esplicitamente, a mano o con questo flag, mai in automatico.
# ---------------------------------------------------------------------
PURGE_SHARDS=0
PURGE_LEGACY_MODE=0
PURGE_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --purge-shards) PURGE_SHARDS=1 ;;
    --purge-legacy-mode) PURGE_LEGACY_MODE=1 ;;
    --purge-models) PURGE_MODELS=1 ;;
  esac
done

wait_for_service_removal() {
  local svc="$1"
  local max_attempts=60   # 60 tentativi * 10s = 10 minuti di margine
  local attempt=0

  echo "    Attendo lo spegnimento reale dei task di '$svc'..."
  while [ "$attempt" -lt "$max_attempts" ]; do
    local status running_count
    status=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$svc" \
      --query "services[0].status" --output text --region "$REGION" 2>/dev/null || echo "MISSING")

    if [ "$status" == "MISSING" ] || [ "$status" == "None" ] || [ "$status" == "INACTIVE" ]; then
      echo "    '$svc' completamente rimosso (status: $status)."
      return 0
    fi

    running_count=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$svc" \
      --query "services[0].runningCount" --output text --region "$REGION" 2>/dev/null || echo "0")

    if [ "$running_count" == "0" ] || [ "$running_count" == "None" ]; then
      echo "    '$svc' a runningCount=0 (status: $status)."
      return 0
    fi

    sleep 10
    attempt=$((attempt + 1))
  done

  echo "    [ATTENZIONE] Timeout in attesa dello spegnimento di '$svc' (ancora presenti task dopo 10 minuti)."
  echo "                 Verifica manualmente con: aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $svc --region $REGION"
  return 0  # non blocchiamo l'intero teardown per un singolo service lento
}

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
    echo "    Eliminazione definitiva del servizio '$svc'..."
    aws ecs update-service --cluster "$CLUSTER_NAME" --service "$svc" --desired-count 0 --region "$REGION" > /dev/null 2>&1
    aws ecs delete-service --cluster "$CLUSTER_NAME" --service "$svc" --force --region "$REGION" > /dev/null 2>&1 \
      && echo "    '$svc' eliminato con successo." \
      || echo "    ($svc: eliminazione fallita, salto)"
  done

  echo ""
  echo "==> SINCRONIZZAZIONE AWS: Attendo la distruzione di TUTTI i container attivi..."
  echo "    (Fargate sta spegnendo i vecchi nodi zombie. Il terminale si sbloccherà automaticamente, attendi...)"

  # Polling esplicito per-service invece del waiter 'services-stable' (che su
  # service già eliminati fallisce sempre, vedi commento di wait_for_service_removal).
  echo "==> [2/5] Attendo lo spegnimento REALE dei container (Sincronizzazione)..."
  for svc in "${TARGET_SERVICES[@]}"; do
    wait_for_service_removal "$svc"
  done
  echo "    Fargate ha spento tutti i container (o il timeout di sicurezza è scaduto, vedi eventuali avvisi sopra)."

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
echo "    Tabelle: workers_registry, orchestrators_registry, JobLocks, ModelStatus, OrchestratorLocks, WorkerTasks, WorkerIndexLocks, JobMetadata"
echo "    Nota: vengono rimossi solo gli ITEM, le tabelle restano intatte."

# Elenco confermato dalla console DynamoDB (8 tabelle totali usate dal sistema)
# NOTA: il nome corretto e' "JobMetadata" (case-sensitive) - la tabella creata a
# mano come "JobMetaData" era un refuso ormai eliminato, vedi validazione failover.
DYNAMO_TABLES=(
  "workers_registry"
  "orchestrators_registry"
  "JobLocks"
  "ModelStatus"
  "OrchestratorLocks"
  "WorkerTasks"
  "WorkerIndexLocks"
  "JobMetadata"
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

# ---------------------------------------------------------------------
# BUG CORRETTO: il vecchio step [5/5] puliva solo "temp/", un prefisso
# che in realtà non esiste mai nel bucket (verificato dalla console S3):
# era quindi un no-op silenzioso, e i veri artefatti di test (job con
# timestamp nel nome, mai sovrascritti) si accumulavano indefinitamente.
#
# Prefissi PULITI ad ogni teardown (dati temporanei/di run, rigenerabili
# automaticamente al prossimo test):
#   - distributed_trains/  (chunk di training caricati per il job)
#   - distributed_tests/   (chunk di test caricati per il job)
#   - tasks/                (stato/metadati dei singoli task RPC)
#   - checkpoints/          (normalmente auto-pulito a fine job riuscito,
#                            qui ripuliamo eventuali orfani da job falliti/interrotti)
#
# Prefissi SALVAGUARDATI di default (mai toccati da questo script):
#   - real/                 dataset sorgente
#   - federated_shards/     shard federati (rigenerarli richiede risplittare
#                            il dataset da zero: usa --purge-shards se serve)
#   - federated_config/     configurazione dello split federato
#   - saved_models/         output vero e proprio dei training (modelli .pkl):
#                            cancellazione solo esplicita, a mano o con --purge-models
#   - metrics/, test_reports/  risultati dei test, servono per i confronti
# ---------------------------------------------------------------------
echo "==> [5/5] Pulizia degli artefatti temporanei di test su S3..."
for prefix in "distributed_trains/" "distributed_tests/" "tasks/" "checkpoints/"; do
  aws s3 rm "s3://$DATASETS_BUCKET_NAME/$prefix" --recursive --region "$REGION" > /dev/null 2>&1 \
    && echo "    Ripulito: $prefix" \
    || echo "    (${prefix}: già vuoto o non presente)"
done

if [ "$PURGE_SHARDS" -eq 1 ]; then
  echo "    --purge-shards attivo: rimuovo anche federated_shards/ e federated_config/..."
  aws s3 rm "s3://$DATASETS_BUCKET_NAME/federated_shards/" --recursive --region "$REGION" > /dev/null 2>&1 \
    && echo "    Ripulito: federated_shards/" \
    || echo "    (federated_shards/: già vuoto o non presente)"
  aws s3 rm "s3://$DATASETS_BUCKET_NAME/federated_config/" --recursive --region "$REGION" > /dev/null 2>&1 \
    && echo "    Ripulito: federated_config/" \
    || echo "    (federated_config/: già vuoto o non presente)"
fi

if [ "$PURGE_MODELS" -eq 1 ]; then
  echo "    --purge-models attivo: rimuovo anche saved_models/..."
  aws s3 rm "s3://$DATASETS_BUCKET_NAME/saved_models/" --recursive --region "$REGION" > /dev/null 2>&1 \
    && echo "    Ripulito: saved_models/" \
    || echo "    (saved_models/: già vuoto o non presente)"
fi

echo "    Salvaguardati (default): real/, federated_shards/, federated_config/, saved_models/, metrics/, test_reports/."
echo "==> [6/6] Deregistrazione di tutte le revision delle task definition (rf-*)..."
FAMILIES=$(aws ecs list-task-definition-families --region "$REGION" --status ACTIVE \
  --query "families[?starts_with(@, 'rf-')]" --output text 2>/dev/null || echo "")

if [ -z "$FAMILIES" ]; then
  echo "    Nessuna famiglia 'rf-*' trovata."
else
  for family in $FAMILIES; do
    ARNS=$(aws ecs list-task-definitions --family-prefix "$family" --region "$REGION" \
      --query "taskDefinitionArns" --output text 2>/dev/null || echo "")
    count=0
    for arn in $ARNS; do
      aws ecs deregister-task-definition --task-definition "$arn" --region "$REGION" > /dev/null 2>&1
      count=$((count + 1))
    done
    [ "$count" -gt 0 ] && echo "    $family: $count revision deregistrate."
  done
fi
echo ""
echo "========================================================================"
echo " PULIZIA E RESUBMISSION REALE COMPLETATE"
echo "========================================================================"
echo "Fatto! AWS Fargate ha rimosso con successo ogni container dal cluster."
echo "Le tabelle DynamoDB (workers/orchestrators/job) sono state svuotate: il prossimo"
echo "deploy riparte da una tela pulita, senza job o registrazioni fantasma."
echo "I dataset in 'real/' e i modelli in 'saved_models/' sono al sicuro (salvo --purge-models)."
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
    echo "Per riaverli, cambia training_mode in terraform.tfvars e rilancia terraform apply (li ricrea da zero)."
  fi
fi