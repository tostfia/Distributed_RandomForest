#!/bin/bash

# Prende la cartella in cui risiede questo script e si sposta lì
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Definisce i percorsi relativi alla cartella dello script
TARGET_DIRS=(
    "$SCRIPT_DIR/.local_storage"
    "$SCRIPT_DIR/saved_models"
)

echo "[CLEANUP] Avvio pulizia delle cartelle di progetto..."

for DIR in "${TARGET_DIRS[@]}"; do
    # Verifica se la cartella esiste
    if [ ! -d "$DIR" ]; then
        echo "[WARNING] La cartella non esiste: $DIR"
        continue
    fi

    echo "Pulizia della cartella: $DIR"

    # Abilitiamo il "dotglob" per includere i file nascosti (che iniziano con .) nel ciclo
    shopt -s dotglob
    
    for FILE in "$DIR"/*; do
        # Verifica se la cartella è vuota
        [ -e "$FILE" ] || continue
        
        # Estrae solo il nome del file/cartella dal path assoluto
        BASENAME=$(basename "$FILE")
        
        # Salta i puntatori di directory correnti e superiori
        if [ "$BASENAME" = "." ] || [ "$BASENAME" = ".." ]; then
            continue
        fi

        # SE il file è .gitkeep, lo ignoriamo e lo conserviamo
        if [ "$BASENAME" = ".gitkeep" ]; then
            echo "Conservato: .gitkeep"
            continue
        fi

        # Eliminazione effettiva
        if [ -d "$FILE" ] && [ ! -L "$FILE" ]; then
            # Se è una cartella reale, rimuovila ricorsivamente
            rm -rf "$FILE"
            echo " Eliminata cartella e contenuto: $BASENAME"
        else
            # Se è un file o un link simbolico, rimuovilo
            rm -f "$FILE"
            echo "Eliminato file: $BASENAME"
        fi
    done
    
    # Ripristiniamo l'opzione di default di bash per sicurezza
    shopt -u dotglob
done

echo -e "[CLEANUP] Pulizia completata!\n"