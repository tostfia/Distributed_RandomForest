#!/bin/bash

# Prende la cartella in cui risiede questo script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Definisce i percorsi
TARGET_DIRS=(
    "$SCRIPT_DIR/.local_storage"
    "$SCRIPT_DIR/saved_models"
    "$SCRIPT_DIR/workers_cache"

)

echo "[CLEANUP] Avvio pulizia delle cartelle di progetto..."

for DIR in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "[WARNING] La cartella non esiste o non è un percorso valido: $DIR"
        continue
    fi

    echo "Pulizia in corso nella cartella: $DIR"

    find "$DIR" -maxdepth 1 \
        ! -name "." \
        ! -name ".gitkeep" \
        -exec rm -rf {} +

    echo " [OK] Contenuto di $DIR pulito (tranne .gitkeep)."
done

echo -e "[CLEANUP] Pulizia completata!\n"