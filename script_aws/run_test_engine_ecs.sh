#!/bin/bash
set -e

# =====================================================================
# RUN TEST ENGINE (task diretto, non interattivo): lancia il TestEngine
# DENTRO la VPC come task ECS one-off, con lo scenario da eseguire passato
# come variabile d'ambiente SCENARIO — NON più via sessione interattiva
# 'aws ecs execute-command'.
#
# Perché questo cambio: ECS Exec ha un timeout di inattività FISSO a 20
# minuti (non configurabile, vedi AWS docs), misurato sull'input da
# tastiera del client — non sull'output che scorre. Test lunghi (training
# federato, scenario di scalabilità con più configurazioni di worker)
# eccedono facilmente quella finestra, troncando la sessione a metà.
# Facendo girare l'engine come comando PRINCIPALE del task (esattamente
# come già fanno worker-service/orchestrator-service), il lavoro non
# dipende più da nessuna sessione client: il task Fargate può girare per
# ore, e i suoi log finiscono su CloudWatch come qualunque altro
# container — cosa che PRIMA non succedeva (l'output dell'engine, iniettato
# via execute-command, esisteva solo nel canale SSM finché restavi
# collegata, senza persistenza server-side).
#
# Prerequisiti:
#   - deploy.sh già eseguito con successo (worker-service/orchestrator-service
#     RUNNING sul cluster 'forest-cluster')
#   - terraform apply già eseguito: la Task Definition 'rf-test-engine-task'
#     è gestita in ecs_task_definitions.tf (risorsa aws_ecs_task_definition.
#     test_engine), NON più registrata a mano da questo script. Qui la
#     usiamo così com'è e sovrascriviamo solo SCENARIO a runtime via
#     'run-task --overrides'.
#   - .env con ENV_MODE=aws e credenziali AWS Academy correnti
#   - engine.py deve leggere la scelta da os.environ.get("SCENARIO") prima
#     del prompt interattivo (già presente in src/testing/engine.py)
#
# Uso:
#   ./run_test_engine_ecs.sh <scenario>      # es. ./run_test_engine_ecs.sh 2
#   ./run_test_engine_ecs.sh                 # chiede lo scenario a terminale
#                                             # PRIMA di lanciare il task
#                                             # (nessuna sessione ECS coinvolta)
# =====================================================================

if [ ! -f .env ]; then
  echo "[ERRORE] File .env non trovato nella directory corrente ($(pwd))."
  exit 1
fi

get_env_var() {
  local key="$1"
  grep -E "^[[:space:]]*${key}[[:space:]]*=" .env | cut -d '=' -f 2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}

ENV_ENV_MODE=$(get_env_var "ENV_MODE")
ENV_TRAINING_MODE=$(get_env_var "TRAINING_MODE")
ENV_NUM_WORKERS=$(get_env_var "NUM_WORKERS")
ENV_REGION=$(get_env_var "AWS_DEFAULT_REGION")
ENV_BUCKET_NAME=$(get_env_var "DATASETS_BUCKET_NAME")
ENV_RPC_SYNC_TIMEOUT=$(get_env_var "RPC_SYNC_TIMEOUT_SECONDS")
ENV_RPC_INFERENCE_SYNC_TIMEOUT=$(get_env_var "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS")
RPC_SYNC_TIMEOUT_SECONDS="${ENV_RPC_SYNC_TIMEOUT:-1800}"
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS="${ENV_RPC_INFERENCE_SYNC_TIMEOUT:-900}"

if [ "$ENV_ENV_MODE" != "aws" ]; then
  echo "[ERRORE] ENV_MODE nel .env è '$ENV_ENV_MODE', non 'aws'."
  exit 1
fi

TRAINING_MODE="${ENV_TRAINING_MODE:-centralized}"
REGION="${ENV_REGION:-us-east-1}"
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"
BUCKET_NAME="${ENV_BUCKET_NAME:-my-cluster-datasets-bucket-759804778194-us-east-1-an}"
CLUSTER_NAME="forest-cluster"
SG_NAME="rf-distributed-sg"
FAMILY="rf-test-engine-task"
CONTAINER_NAME="test-engine"

if [[ "$TRAINING_MODE" != "centralized" && "$TRAINING_MODE" != "federated" ]]; then
  echo "ERRORE: TRAINING_MODE deve essere 'centralized' o 'federated', ricevuto: '$TRAINING_MODE'"
  exit 1
fi

# ---------------------------------------------------------------------
# Scelta dello scenario: primo argomento posizionale, oppure prompt
# locale (nessuna sessione ECS coinvolta — puoi rispondere con calma,
# non consuma nessun timeout).
# ---------------------------------------------------------------------
VALID_SCENARIOS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "all")
SCENARIO_CHOICE="$1"

is_valid_scenario() {
  local candidate="$1"
  for v in "${VALID_SCENARIOS[@]}"; do
    [ "$v" == "$candidate" ] && return 0
  done
  return 1
}

if [ -z "$SCENARIO_CHOICE" ]; then
  echo "Seleziona lo scenario da eseguire:"
  echo "1. Performance e Metriche"
  echo "2. Scalabilità"
  echo "3. Simulazione di Rete"
  echo "4. Guasto improvviso del Worker (addestramento)"
  echo "5. Guasto improvviso del Worker (inferenza)"
  echo "6. Failover dell'Orchestratore (addestramento)"
  echo "7. Failover dell'Orchestratore (inferenza)"
  echo "8. Elezione del Leader sotto Concorrenza (Safety)"
  echo "9. Genera Grafici"
  while true; do
    read -p "Scelta (1-9, o 'all' per eseguire tutti): " SCENARIO_CHOICE
    SCENARIO_CHOICE="$(echo "$SCENARIO_CHOICE" | tr '[:upper:]' '[:lower:]' | xargs)"
    is_valid_scenario "$SCENARIO_CHOICE" && break
    echo "[ERRORE] Opzione non valida. Riprova."
  done
else
  SCENARIO_CHOICE="$(echo "$SCENARIO_CHOICE" | tr '[:upper:]' '[:lower:]' | xargs)"
  if ! is_valid_scenario "$SCENARIO_CHOICE"; then
    echo "[ERRORE] Scenario '$SCENARIO_CHOICE' non valido. Valori ammessi: ${VALID_SCENARIOS[*]}"
    exit 1
  fi
fi

# ---------------------------------------------------------------------
# [SAFETY NET] Questo trap protegge SOLO la fase di setup (step 1-4, prima
# che il task venga lanciato): se qualcosa fallisce lì (credenziali,
# servizi non stabili, VPC/SG mancanti), 'set -e' esce e non c'è nessun
# task da fermare (TASK_ARN è ancora vuoto, il trap non fa nulla).
# Una volta che il task è REALMENTE partito con lo scenario scelto, il
# trap viene disattivato esplicitamente (vedi fondo script): a differenza
# della vecchia versione (dove il task faceva solo 'sleep infinity' in
# attesa di te, e lasciarlo acceso per errore sprecava Fargate), ora il
# task sta facendo il lavoro che hai chiesto — non va fermato solo perché
# lo script termina o perdi la connessione locale.
# ---------------------------------------------------------------------
TASK_ARN=""

cleanup_on_setup_failure() {
  local exit_code=$?
  if [ -n "$TASK_ARN" ]; then
    echo ""
    echo "[CLEANUP] Fermo il task del test-engine ($TASK_ARN), lanciato ma non ancora confermato RUNNING..."
    aws ecs stop-task --cluster "$CLUSTER_NAME" --task "$TASK_ARN" --region "$REGION" > /dev/null 2>&1 \
      && echo "[CLEANUP] Task fermato." \
      || echo "[CLEANUP] (task già fermo, non trovato, o stop già eseguito: ok)"
  fi
  exit $exit_code
}
trap cleanup_on_setup_failure EXIT INT TERM HUP

echo "===================================================================="
echo " RUN TEST ENGINE (task diretto)  -  forest-cluster ($REGION)"
echo "===================================================================="
echo " TRAINING_MODE : $TRAINING_MODE"
echo " NUM_WORKERS   : $NUM_WORKERS"
echo " SCENARIO      : $SCENARIO_CHOICE"
echo "--------------------------------------------------------------------"

echo "==> [1/5] Verifica credenziali AWS (Learner Lab, scadono ogni ~4h)..."
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute. Aggiornale nel .env e riprova."
  exit 1
fi
echo "    OK, credenziali valide."

echo "==> [2/5] Verifica che worker-service/orchestrator-service siano stabili..."
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
  echo "[ERRORE] Nessun Service worker/orchestrator trovato sul cluster '$CLUSTER_NAME'. Hai lanciato deploy.sh?"
  exit 1
fi
echo "    Service trovati: ${TARGET_SERVICES[*]}"

# L'API ECS (sia 'describe-services' sia il waiter 'services-stable' che la
# usa sotto) accetta al massimo 10 nomi di servizio per chiamata. Con
# NUM_WORKERS alto + orchestrator-service si supera facilmente il limite,
# quindi spezziamo la lista in chunk da 10 e aspettiamo ogni chunk
# separatamente invece che tutti insieme in una singola chiamata.
CHUNK_SIZE=10
TOTAL_SERVICES="${#TARGET_SERVICES[@]}"
for ((i=0; i<TOTAL_SERVICES; i+=CHUNK_SIZE)); do
  CHUNK=("${TARGET_SERVICES[@]:i:CHUNK_SIZE}")
  echo "    Attendo stabilità per: ${CHUNK[*]}"
  aws ecs wait services-stable --cluster "$CLUSTER_NAME" --services "${CHUNK[@]}" --region "$REGION"
done
echo "    OK, infrastruttura pronta."

echo "==> [3/5] Recupero VPC/subnet/Security Group (stessi usati da deploy.sh)..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query "Subnets[*].SubnetId" --output text --region "$REGION")
SUBNET_ARRAY=($SUBNET_IDS)
SUBNET_1=${SUBNET_ARRAY[0]}
SUBNET_2=${SUBNET_ARRAY[1]:-${SUBNET_ARRAY[0]}}
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION")
if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
  echo "[ERRORE] Security Group '$SG_NAME' non trovato: hai lanciato deploy.sh?"
  exit 1
fi
echo "    VPC: $VPC_ID | Subnet: $SUBNET_1, $SUBNET_2 | SG: $SG_ID"
echo "    (Stesso SG dei worker: la regola self-referencing sulla porta 18861"
echo "     già presente basta, il test-engine sarà raggiungibile dai worker e"
echo "     viceversa senza aprire nulla verso l'esterno.)"

echo "==> [4/5] Verifica Task Definition '$FAMILY' (gestita da Terraform)..."
# La Task Definition non viene più registrata a mano da questo script:
# vive in ecs_task_definitions.tf (risorsa aws_ecs_task_definition.
# test_engine) ed è creata/aggiornata da 'terraform apply', come le
# altre (orchestrator, worker). Qui verifichiamo solo che esista già,
# per dare un errore chiaro invece di un fallimento oscuro in run-task.
if ! aws ecs describe-task-definition --task-definition "$FAMILY" --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Task Definition '$FAMILY' non trovata. Hai lanciato 'terraform apply'?"
  exit 1
fi
echo "    OK, Task Definition trovata."

echo "==> [5/5] Avvio del task (run-task, one-off)..."
# Niente più '--enable-execute-command': non entriamo più nel container
# con una sessione interattiva, quindi non serve l'agente ECS Exec.
# SCENARIO viene sovrascritto qui via --overrides: la Task Definition di
# base (Terraform) ha un valore di default, ma questo è quello che conta
# davvero, deciso a runtime da chi lancia lo script.
CONTAINER_OVERRIDES=$(cat <<EOF
{
  "containerOverrides": [
    {
      "name": "${CONTAINER_NAME}",
      "environment": [
        {"name": "NUM_WORKERS", "value": "${NUM_WORKERS}"},
        {"name": "TRAINING_MODE", "value": "${TRAINING_MODE}"},
        {"name": "DATASETS_BUCKET_NAME", "value": "${BUCKET_NAME}"},
        {"name": "RPC_SYNC_TIMEOUT_SECONDS", "value": "${RPC_SYNC_TIMEOUT_SECONDS}"},
        {"name": "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", "value": "${RPC_INFERENCE_SYNC_TIMEOUT_SECONDS}"},
        {"name": "SCENARIO", "value": "${SCENARIO_CHOICE}"}
      ]
    }
  ]
}
EOF
)
TASK_ARN=$(aws ecs run-task \
  --cluster "$CLUSTER_NAME" \
  --task-definition "$FAMILY" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --overrides "$CONTAINER_OVERRIDES" \
  --region "$REGION" \
  --query "tasks[0].taskArn" --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" == "None" ]; then
  echo "[ERRORE] run-task non ha restituito un Task ARN valido."
  exit 1
fi
echo "    Task avviato: $TASK_ARN"

echo "    Attendo che il task sia RUNNING..."
for attempt in $(seq 1 30); do
  TASK_STATUS=$(aws ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$TASK_ARN" \
    --query "tasks[0].lastStatus" --output text --region "$REGION")

  if [ "$TASK_STATUS" == "RUNNING" ]; then
    echo "    OK: task RUNNING."
    break
  fi
  sleep 5
  if [ "$attempt" -eq 30 ]; then
    echo "[ERRORE] Timeout (150s) in attesa che il task sia RUNNING."
    echo "         Stato task: $TASK_STATUS"
    echo "         Puoi controllare manualmente con:"
    echo "           aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --region $REGION"
    exit 1
  fi
done

# Il task è confermato RUNNING con lo scenario scelto: da qui in avanti
# lavora da solo, indipendentemente da questo script/terminale. Disattivo
# il trap di sicurezza — non deve più fermarlo all'uscita.
trap - EXIT INT TERM HUP

echo ""
echo "===================================================================="
echo " Task avviato con successo. Il lavoro prosegue in background sul"
echo " cluster, indipendentemente da questo terminale — puoi anche"
echo " chiuderlo, il task NON verrà fermato."
echo "===================================================================="
echo " Per seguire i log in tempo reale:"
echo "   aws logs tail /ecs/rf-test-engine --follow --region $REGION"
echo ""
echo " Per controllare se il task ha terminato:"
echo "   aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN \\"
echo "     --query \"tasks[0].lastStatus\" --region $REGION"
echo ""
echo " Il report finale (quando lo scenario termina) viene caricato in:"
echo "   s3://$BUCKET_NAME/test_reports/aws/"
echo ""
echo " Per fermarlo manualmente prima che finisca:"
echo "   aws ecs stop-task --cluster $CLUSTER_NAME --task $TASK_ARN --region $REGION"
echo "===================================================================="