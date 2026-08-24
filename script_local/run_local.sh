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

# ---------------------------------------------------------------------
# Caricamento del .env
# ---------------------------------------------------------------------
#
# VERSIONE PRECEDENTE:
#
#     export "$(echo "$line" | tr -d ' ')"
#
# 'tr -d " "' cancella TUTTI gli spazi della riga, non solo quelli attorno
# all'uguale. Conseguenze concrete su questo progetto:
#
#   1) qualunque valore contenente spazi veniva silenziosamente corrotto. Il
#      percorso del progetto stesso ne contiene uno
#      ("~/Progetto SDCC-ML/Distributed_RandomForest"), quindi una riga come
#
#          DATASET_LOCAL_PATH=/home/gaia/Progetto SDCC-ML/dataset_cache
#
#      diventava ".../ProgettoSDCC-ML/dataset_cache": una cartella che non
#      esiste, con un FileNotFoundError che sembra un errore di battitura
#      dell'utente e non dello script;
#
#   2) le virgolette non venivano rimosse: VAR="valore" esportava il valore
#      CON gli apici, e i confronti in Python fallivano contro la stringa nuda;
#
#   3) le righe con 'export ' davanti, i commenti indentati e le terminazioni
#      di riga Windows (CRLF) non erano gestiti.
#
# Qui chiave e valore vengono separati al PRIMO '=' e trattati in modo diverso:
# dalla chiave gli spazi si tolgono davvero (una variabile d'ambiente non può
# contenerne), dal valore si tolgono solo quelli ai bordi. Il contenuto resta
# intatto.
# ---------------------------------------------------------------------
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Terminazioni di riga Windows: senza questo, l'ultimo carattere del
        # valore sarebbe un \r invisibile.
        line="${line%$'\r'}"

        # Commenti, anche indentati (la versione precedente riconosceva solo
        # quelli che iniziavano a colonna 1).
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        # Righe vuote o composte da soli spazi.
        [[ -z "${line//[[:space:]]/}" ]] && continue

        # Righe senza '=': non sono assegnazioni, si ignorano invece di
        # esportare qualcosa di malformato.
        [[ "$line" != *=* ]] && continue

        # Prefisso 'export ' opzionale, dopo aver tolto l'indentazione.
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line#export }"

        key="${line%%=*}"
        value="${line#*=}"

        # La CHIAVE non può contenere spazi: qui toglierli è corretto.
        key="${key//[[:space:]]/}"
        [[ -z "$key" ]] && continue

        # Dal VALORE si tolgono solo gli spazi ai bordi.
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        # Rimozione di UNA coppia di apici esterni, se presente.
        if [[ ${#value} -ge 2 && "$value" == \"*\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ ${#value} -ge 2 && "$value" == \'*\' ]]; then
            value="${value:1:${#value}-2}"
        fi

        export "$key=$value"
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

if [ ! -f "$(pwd)/worker_supervisor.py" ]; then
    echo "[ERRORE] worker_supervisor.py non trovato nella root del progetto ($(pwd))."
    echo "         Serve per il restart-on-failure automatico dei worker in locale."
    exit 1
fi

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
    $TERM_CMD bash -c "export PYTHONPATH=\"${ROOT_DIR}:${ROOT_DIR}/src\"; export ORCHESTRATOR_INDEX=$i; python -m src.master.orchestrator.main; exec bash"
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
    $TERM_CMD bash -c "export PYTHONPATH=\"${ROOT_DIR}:${ROOT_DIR}/src\"; python \"${ROOT_DIR}/worker_supervisor.py\" -- python -m src.worker.main $WORKER_NAME $PORT ; exec bash"
done

sleep 2

# 3. Avvio del Client direttamente nel terminale corrente
echo "[START] Avvio Client Interattivo..."
# Modificato: include anche la cartella /src nel PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"
python -m src.client.main

# Esegue la pulizia finale della rete
cleanup