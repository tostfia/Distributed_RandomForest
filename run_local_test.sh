#!/bin/bash

# Funzione di pulizia di emergenza per evitare processi zombie in locale
cleanup() {
    echo -e "\n[CLEANUP LOCAL] Terminazione forzata di eventuali supervisor o worker residui..."
    pkill -f "worker_supervisor.py"
    pkill -f "src.worker.main"
    exit
}

# Cattura interruzioni improvvise (Ctrl+C)
trap cleanup SIGINT SIGTERM

ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"


echo "[SYSTEM] Lancio del Test Engine in ambiente Locale..."

# Avvia l'engine (che internamente tirerà su i worker con il supervisor)
python -m src.testing.engine

# Al termine dei test, esegue una pulizia di sicurezza
cleanup