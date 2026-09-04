#!/bin/bash

# Equivalente Docker di run_local.sh: stesso flusso (parsing .env, scelta
# numero orchestratori, provisioning automatico, poi avvio del cluster),
# ma orchestrator/worker girano in container con limiti di CPU/RAM (vedi
# docker-compose.yml: cpus/mem_limit) invece che come processi nativi
# illimitati sull'host -- pensato per non saturare la macchina di sviluppo
# (VS Code compreso) quando NUM_WORKERS è alto.
#
# NOTA: il client resta sull'host (non containerizzato, nessun servizio
# 'client' in docker-compose.yml): comunica con l'orchestratore containerizzato
# tramite la coda SQS mock su disco (./.local_storage, montata sia sull'host
# sia nei container), non via RPC diretta -- nessun problema di rete lì.

cleanup() {
    echo -e "\n[CLEANUP] Arresto e rimozione dei container (docker-compose down)..."
    docker-compose down
    echo "[CLEANUP] Fatto."
    exit
}

trap cleanup SIGINT SIGTERM

MODE=$1
if [ "$MODE" != "delay" ] && [ "$MODE" != "puro" ]; then
    echo "Uso corretto: ./run_docker.sh [delay|puro]"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "[ERRORE] Né 'docker-compose' né 'docker compose' risultano disponibili."
    exit 1
fi
# Preferisce 'docker compose' (plugin, sintassi nuova) se disponibile, altrimenti
# ricade sul binario standalone 'docker-compose' (sintassi vecchia, con trattino).
if docker compose version &> /dev/null; then
    DC="docker compose"
else
    DC="docker-compose"
fi

echo -n "Quanti nodi Orchestratore vuoi avviare (1-2)? "
read NUM_ORCHESTRATORS

if ! [[ "$NUM_ORCHESTRATORS" =~ ^[1-2]$ ]]; then
    echo "Errore: inserisci un numero compreso tra 1 e 2."
    exit 1
fi

# ---------------------------------------------------------------------
# Caricamento del .env -- STESSO parser robusto di run_local.sh (vedi lì
# per la motivazione dettagliata: separazione chiave/valore al primo '=',
# gestione spazi/apici/CRLF/commenti indentati). Duplicato qui invece che
# fattorizzato in un file comune per restare un singolo script autonomo,
# copiabile/eseguibile senza dipendenze aggiuntive -- se lo modifichi in
# uno dei due script, aggiornalo nell'altro.
# ---------------------------------------------------------------------
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        [[ "$line" != *=* ]] && continue
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line#export }"
        key="${line%%=*}"
        value="${line#*=}"
        key="${key//[[:space:]]/}"
        [[ -z "$key" ]] && continue
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
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

if [ -z "$NUM_WORKERS" ]; then
    echo "[ERRORE] NUM_WORKERS non definito nel file .env"
    exit 1
fi

echo "[SYSTEM] Rilevati $NUM_WORKERS worker dal file .env"

ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"

# ---------------------------------------------------------------------
# Provisioning automatico degli shard federati -- STESSO passo di
# run_local.sh, eseguito qui sull'HOST (non dentro un container): scrive
# su ./workers_cache, che è la stessa cartella montata nei container worker
# (vedi docker-compose.yml: "./workers_cache:/app/workers_cache:z"), quindi
# gli shard prodotti qui sono immediatamente visibili anche da dentro i
# container, senza bisogno di rifare il provisioning containerizzato.
# ---------------------------------------------------------------------
if [ "${TRAINING_MODE:-centralized}" = "federated" ]; then
    echo "[PROVISIONING] TRAINING_MODE=federated rilevato: verifico/preparo gli shard federati..."
    python -m script_local.provision_local_shards
    PROVISION_EXIT=$?
    if [ $PROVISION_EXIT -ne 0 ]; then
        echo "[ERRORE] Provisioning degli shard federati fallito (exit $PROVISION_EXIT)."
        echo "         Correggi l'errore sopra e rilancia -- il cluster NON viene avviato."
        exit 1
    fi
    echo "[PROVISIONING OK] Shard federati pronti."
fi

# ---------------------------------------------------------------------
# Scenario di rete -- a differenza di run_local.sh (che usa 'tc' sull'host,
# su 'lo'), qui il ritardo va applicato DENTRO al container worker (comando
# 'tc qdisc add dev eth0 root netem $NET_SCENARIO' già presente nel
# 'command:' del servizio worker in docker-compose.yml, richiede
# cap_add: NET_ADMIN -- già presente lì). Impostiamo qui solo la variabile
# NET_SCENARIO che quel comando legge.
# ---------------------------------------------------------------------
if [ "$MODE" = "delay" ]; then
    export NET_SCENARIO="delay 50ms"
    echo "[RETENET] I worker applicheranno 50ms di latenza in ingresso (via 'tc' nel container)."
else
    # "delay 0ms" invece di lasciare NET_SCENARIO vuoto: un parametro vuoto
    # dopo 'netem' nel comando del worker (vedi docker-compose.yml) farebbe
    # fallire silenziosamente 'tc' con un warning ad ogni avvio worker --
    # 'delay 0ms' è un comando 'tc' valido che semplicemente non introduce
    # alcun ritardo, quindi nessun warning spurio nei log.
    export NET_SCENARIO="delay 0ms"
    echo "[RETENET] Nessun ritardo di rete aggiuntivo (modalità 'puro')."
fi

echo "[SYSTEM] Build dell'immagine (se necessario)..."
$DC build orchestrator
if [ $? -ne 0 ]; then
    echo "[ERRORE] Build dell'immagine Docker fallita."
    exit 1
fi

echo "[SYSTEM] Avvio del cluster: $NUM_ORCHESTRATORS orchestratore/i, $NUM_WORKERS worker..."
$DC up -d --scale orchestrator=$NUM_ORCHESTRATORS --scale worker=$NUM_WORKERS orchestrator worker
if [ $? -ne 0 ]; then
    echo "[ERRORE] Avvio dei container fallito. Controlla 'docker compose logs' per i dettagli."
    exit 1
fi

echo ""
echo "[SYSTEM] Cluster avviato in background (container, con limiti CPU/RAM da .env)."
echo "         Per seguire i log in una finestra separata:"
echo "             $DC logs -f"
echo "         Per lo stato dei container:"
echo "             $DC ps"
echo ""

# Il client resta sull'host (nessun servizio 'client' in docker-compose.yml):
# comunica con l'orchestratore containerizzato tramite la coda SQS mock su
# disco (./.local_storage, montata sia sull'host sia nei container), non via
# RPC diretta -- nessuna configurazione di rete aggiuntiva necessaria qui.
echo "[START] Avvio Client Interattivo (sull'host)..."
python -m src.client.main

cleanup