#!/bin/bash

# Prende la cartella in cui risiede questo script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Definisce i percorsi
TARGET_DIRS=(
    "$SCRIPT_DIR/.local_storage/"
    "$SCRIPT_DIR/saved_models/"
    "$SCRIPT_DIR/workers_cache/"
)

echo "[CLEANUP] Avvio pulizia selettiva dei contenuti..."

for DIR in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "[WARNING] La cartella non esiste o non è un percorso valido: $DIR"
        continue
    fi

    echo "Svuotamento dei contenuti in: $DIR"

    # MODIFICA: 'mindepth 1' evita di toccare la cartella radice.
    # L'esclusione di .gitkeep protegge il file di placeholder.
    find "$DIR" -mindepth 1 \
        ! -name ".gitkeep" \
        -exec rm -rf {} +

    echo " [OK] Contenuto di $DIR svuotato (struttura radice e .gitkeep preservati)."
done

echo -e "[CLEANUP] Pulizia completata con successo!\n"