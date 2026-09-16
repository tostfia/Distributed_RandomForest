#!/bin/bash
export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

if [ -f .env ]; then
    ENV_NUM_WORKERS=$(grep -E "^[[:space:]]*NUM_WORKERS[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_TRAINING_MODE=$(grep -E "^[[:space:]]*TRAINING_MODE[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_PARTITION_STRATEGY=$(grep -E "^[[:space:]]*PARTITION_STRATEGY[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_TREE_ALLOCATION_STRATEGY=$(grep -E "^[[:space:]]*TREE_ALLOCATION_STRATEGY[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_DAY_COLUMN=$(grep -E "^[[:space:]]*DAY_COLUMN[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_DATASET_TYPE=$(grep -E "^[[:space:]]*DATASET_TYPE[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_DATASET_LOCAL_PATH=$(grep -E "^[[:space:]]*DATASET_LOCAL_PATH[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
fi
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"
TRAINING_MODE="${ENV_TRAINING_MODE:-centralized}"

echo "[RUN_TEST] Avvio con NUM_WORKERS=$NUM_WORKERS, TRAINING_MODE=$TRAINING_MODE (da .env)..."

# ---------------------------------------------------------------------
# Lo scenario 3 (simulazione di rete) del test engine
# resta l'unico responsabile di introdurre latenza quando richiesto,
# gestendola dinamicamente sui container invece che tramite
# variabile d'ambiente statica.
# ---------------------------------------------------------------------
export NET_SCENARIO="delay 0ms"

# ---------------------------------------------------------------------
if [ "$TRAINING_MODE" = "federated" ]; then

    RESOLVED_DATASET_TYPE="${ENV_DATASET_TYPE:-real}"
    echo "[PROVISIONING] TRAINING_MODE=federated rilevato: preparo gli shard federati (dataset_type=${RESOLVED_DATASET_TYPE})..."


    RESOLVED_PARTITION_STRATEGY="${ENV_PARTITION_STRATEGY:-iid}"
    echo "[PROVISIONING] Strategia di partizionamento: ${RESOLVED_PARTITION_STRATEGY} (da PARTITION_STRATEGY in .env, default 'iid' se assente)"

    # Non consumata dal provisioning (che decide solo come sono fatti gli
    # shard, non come vengono pesati gli alberi in training): esportata qui
    # solo perché arrivi come variabile d'ambiente al container test-engine,
    # dove BaseTestScenario._resolve_federated_partitioning la legge per
    # decidere tree_allocation_strategy nel payload del job (vedi
    # base.py/performance.py). Se docker-compose.yml carica già .env
    # interamente nel container (env_file), questo export è ridondante ma
    # innocuo; se non lo fa, è quello che garantisce che la variabile arrivi.
    export TREE_ALLOCATION_STRATEGY="${ENV_TREE_ALLOCATION_STRATEGY:-proportional}"
    echo "[PROVISIONING] Allocazione alberi: ${TREE_ALLOCATION_STRATEGY} (da TREE_ALLOCATION_STRATEGY in .env, default 'proportional' se assente)"

    PROVISION_ARGS=(--force --num-workers "$NUM_WORKERS" --dataset-type "$RESOLVED_DATASET_TYPE" --partition-strategy "$RESOLVED_PARTITION_STRATEGY")
    if [ -n "$ENV_DATASET_LOCAL_PATH" ]; then
        echo "[PROVISIONING] Cartella dataset locale: ${ENV_DATASET_LOCAL_PATH} (esplicito da .env)"
        PROVISION_ARGS+=(--data-folder "$ENV_DATASET_LOCAL_PATH")
    fi
    if [ "$RESOLVED_PARTITION_STRATEGY" = "by_day" ] && [ -n "$ENV_DAY_COLUMN" ]; then
        echo "[PROVISIONING] Day column: ${ENV_DAY_COLUMN} (esplicito da .env)"
        PROVISION_ARGS+=(--day-column "$ENV_DAY_COLUMN")
    fi

    python -m script_local.provision_local_shards "${PROVISION_ARGS[@]}"
    PROVISION_EXIT=$?
    if [ $PROVISION_EXIT -ne 0 ]; then
        echo "[ERRORE] Provisioning degli shard federati fallito (exit $PROVISION_EXIT)."
        echo "         Correggi l'errore sopra e rilancia -- il test NON viene avviato"
        echo "         senza shard pronti, per evitare worker che crashano in loop al boot."
        exit 1
    fi
    echo "[PROVISIONING OK] Shard federati pronti (strategia: ${RESOLVED_PARTITION_STRATEGY}, rigenerati da zero)."
fi

docker compose down

docker compose up -d --scale worker=$NUM_WORKERS --scale orchestrator=2

docker compose run --rm test-engine

docker compose down