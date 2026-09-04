#!/bin/bash
export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

if [ -f .env ]; then
    ENV_NUM_WORKERS=$(grep -E "^[[:space:]]*NUM_WORKERS[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
    ENV_TRAINING_MODE=$(grep -E "^[[:space:]]*TRAINING_MODE[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
fi
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"
TRAINING_MODE="${ENV_TRAINING_MODE:-centralized}"

echo "[RUN_TEST] Avvio con NUM_WORKERS=$NUM_WORKERS, TRAINING_MODE=$TRAINING_MODE (da .env)..."

# ---------------------------------------------------------------------
# BUG FIX: docker-compose.yml applica un default di "delay 50ms" su
# NET_SCENARIO se la variabile non è esportata (vedi comando del servizio
# 'worker': "tc qdisc add dev eth0 root netem ${NET_SCENARIO:-delay 50ms}").
# run_test.sh non la esportava mai, quindi OGNI test lanciato da qui
# (inclusi performance/scalabilità, non solo lo scenario di rete) partiva
# con 50ms di latenza artificiale già applicata di default, contaminando
# le metriche raccolte.
#
# run_docker.sh già neutralizza questo default in modalità 'puro' con lo
# stesso export (vedi lì per la motivazione: "delay 0ms" è un comando tc
# valido che non introduce alcun ritardo, a differenza di una variabile
# vuota che farebbe fallire silenziosamente 'tc' con un warning). Qui lo
# applichiamo sempre: lo scenario 3 (simulazione di rete) del test engine
# resta l'unico responsabile di introdurre latenza quando richiesto,
# gestendola dinamicamente sui container invece che tramite questa
# variabile d'ambiente statica.
# ---------------------------------------------------------------------
export NET_SCENARIO="delay 0ms"

# ---------------------------------------------------------------------
if [ "$TRAINING_MODE" = "federated" ]; then
    echo "[PROVISIONING] TRAINING_MODE=federated rilevato: verifico/preparo gli shard federati..."
    python -m script_local.provision_local_shards
    PROVISION_EXIT=$?
    if [ $PROVISION_EXIT -ne 0 ]; then
        echo "[ERRORE] Provisioning degli shard federati fallito (exit $PROVISION_EXIT)."
        echo "         Correggi l'errore sopra e rilancia -- il test NON viene avviato"
        echo "         senza shard pronti, per evitare worker che crashano in loop al boot."
        exit 1
    fi
    echo "[PROVISIONING OK] Shard federati pronti."
fi

docker compose down

docker compose up -d --scale worker=$NUM_WORKERS --scale orchestrator=2

docker compose run --rm test-engine

docker compose down