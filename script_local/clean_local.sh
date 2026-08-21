#!/bin/bash

# Prende la cartella in cui risiede questo script (e.g., .../Distributed_RandomForest/script_local)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Risale di un livello per trovare la root del progetto (e.g., .../Distributed_RandomForest)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Definisce i percorsi relativi alla root del progetto
TARGET_DIRS=(
    "$PROJECT_ROOT/.local_storage/"
    "$PROJECT_ROOT/saved_models/"
    "$PROJECT_ROOT/workers_cache/"
    "$PROJECT_ROOT/test_reports/local/"
)

CONFIG_PATH="$PROJECT_ROOT/.local_storage/config.json"
TMP_PRESERVE_PATH="$(mktemp -t baseline_boot_preserve.XXXXXX.json)"
PRESERVE_SCRIPT="$SCRIPT_DIR/preserve_baseline_boot.py"

echo "[CLEANUP] Avvio pulizia selettiva dei contenuti..."

if command -v python3 &> /dev/null; then
    python3 "$PRESERVE_SCRIPT" save "$CONFIG_PATH" "$TMP_PRESERVE_PATH"
else
    echo "[WARNING] python3 non trovato: 'baseline_boot' non verrà preservato in questo cleanup."
fi

for DIR in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "[WARNING] La cartella non esiste o non è un percorso valido: $DIR"
        continue
    fi

    echo "Svuotamento dei contenuti in: $DIR"

        find "$DIR" -mindepth 1 \
        ! -name ".gitkeep" \
        ! -path "$PROJECT_ROOT/.local_storage/metrics" \
        ! -path "$PROJECT_ROOT/.local_storage/metrics/*" \
        -exec rm -rf {} +

    echo "  [OK] Contenuto di $DIR svuotato (struttura radice e .gitkeep preservati)."
done

if command -v python3 &> /dev/null; then
    python3 "$PRESERVE_SCRIPT" restore "$CONFIG_PATH" "$TMP_PRESERVE_PATH"
    if [ -f "$CONFIG_PATH" ]; then
        echo "[CLEANUP] 'baseline_boot' preservata in: $CONFIG_PATH"
    fi
fi

echo -e "[CLEANUP] Pulizia completata con successo!\n"