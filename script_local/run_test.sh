#!/bin/bash
export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

docker compose down

docker compose up -d

docker compose run --rm test-engine

docker compose down