#!/bin/bash
export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

if [ -f .env ]; then
    ENV_NUM_WORKERS=$(grep -E "^[[:space:]]*NUM_WORKERS[[:space:]]*=" .env | cut -d '=' -f 2- | tr -d ' ')
fi
NUM_WORKERS="${ENV_NUM_WORKERS:-2}"

echo "[RUN_TEST] Avvio con NUM_WORKERS=$NUM_WORKERS (da .env)..."

docker compose down
docker compose up -d --scale worker=$NUM_WORKERS --scale orchestrator=2
docker compose run --rm test-engine
docker compose down