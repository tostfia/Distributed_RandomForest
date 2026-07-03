#!/bin/bash

# Funzione di pulizia automatica in caso di interruzione (Ctrl+C) o fine script
cleanup() {
    if [ "$MODE" = "delay" ]; then
        echo -e "\n[CLEANUP] Rimozione dei ritardi di rete da localhost (lo)..."
        sudo tc qdisc del dev lo root 2>/dev/null
    fi
    echo "[CLEANUP] Sistema ripristinato. Nota: Chiudi le finestre rimaste aperte."
    exit
}

trap cleanup SIGINT SIGTERM

MODE=$1
if [ "$MODE" != "delay" ] && [ "$MODE" != "puro" ]; then
    echo "Uso corretto: ./run_local.sh [delay|puro]"
    exit 1
fi

# Rilevamento automatico del terminale su Fedora (Diamo priorità a gnome-terminal appena installato)
if command -v gnome-terminal &> /dev/null; then
    TERM_CMD="gnome-terminal --"
elif command -v gnome-console &> /dev/null; then
    TERM_CMD="gnome-console --"
elif command -v kgx &> /dev/null; then
    TERM_CMD="kgx --"
else
    echo "[WARNING] Nessun emulatore grafico trovato. I processi andranno in background."
    TERM_CMD="bash -c"
fi

echo -n "Quanti nodi Orchestratore vuoi avviare (1-2)? "
read NUM_ORCHESTRATORS

if ! [[ "$NUM_ORCHESTRATORS" =~ ^[1-2]$ ]]; then
    echo "Errore: inserisci un numero compreso tra 1 e 2."
    exit 1
fi

if [ -f .env ]; then
    while read -r line || [ -n "$line" ]; do
        # Ignora commenti e righe vuote
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        export "$(echo "$line" | tr -d ' ')"
    done < .env
else
    echo "[ERRORE] File .env non trovato!"
    exit 1
fi

# 3. Verifica che NUM_WORKERS sia stato letto correttamente
if [ -z "$NUM_WORKERS" ]; then
    echo "[ERRORE] NUM_WORKERS non definito nel file .env"
    exit 1
fi

echo "[SYSTEM] Rilevati $NUM_WORKERS worker dal file .env"

# Configurazione della rete
if [ "$MODE" = "delay" ]; then
    echo "[RETENET] Configurazione ritardo di rete: aggiungo 50ms di latenza su localhost..."
    sudo tc qdisc del dev lo root 2>/dev/null
    sudo tc qdisc add dev lo root netem delay 50ms
fi

echo "[SYSTEM] Avvio del cluster distribuito su terminali differenti..."

# Definiamo la root directory in modo sicuro gestendo gli spazi nel path
ROOT_DIR="$(pwd)"

# 1. Avvio dell'Orchestratore in un nuovo terminale
echo "[START] Avvio di $NUM_ORCHESTRATORS Orchestratore/i Master..."
for ((i=1; i<=NUM_ORCHESTRATORS; i++)); do
    echo "[START] Avvio Istanza Orchestratore #$i..."
    # Modificato: include anche la cartella /src nel PYTHONPATH
    $TERM_CMD bash -c "export PYTHONPATH=\"${ROOT_DIR}:${ROOT_DIR}/src\"; python -m src.master.orchestrator.main; exec bash"
    sleep 1 
done

sleep 2
# 2. Avvio dinamico dei Worker in terminali differenti
PORT_BASE=18861
for ((i=1; i<=NUM_WORKERS; i++)); do
    WORKER_NAME=$(printf "Worker-Locale-%02d" $i)
    PORT=$((PORT_BASE + i - 1))
    echo "[START] Avvio $WORKER_NAME sulla porta $PORT..."
    
    # Modificato: include anche la cartella /src nel PYTHONPATH
    $TERM_CMD bash -c "export PYTHONPATH=\"${ROOT_DIR}:${ROOT_DIR}/src\"; python -m src.worker.main $WORKER_NAME $PORT ; exec bash"
done

sleep 2

# 3. Avvio del Client direttamente nel terminale corrente
echo "[START] Avvio Client Interattivo..."
# Modificato: include anche la cartella /src nel PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"
python -m src.client.main

# Esegue la pulizia finale della rete
cleanup