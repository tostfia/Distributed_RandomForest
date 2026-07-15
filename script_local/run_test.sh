#!/bin/bash
# 1. Ferma tutto quello che c'è di vecchio
docker compose down

# 2. Avvia tutto in background
docker compose up -d

# 3. Lancia il test engine
docker compose run --rm test-engine

# 4. (Opzionale) Pulisci alla fine
docker compose down