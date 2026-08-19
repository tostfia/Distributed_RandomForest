# Distributed Random Forest System

Questo progetto implementa un sistema distribuito per l'addestramento e l'inferenza di modelli Random Forest in modalità Centralizzata e Federata, sviluppato per il progetto congiunto dei corsi di Machine Learning e Sistemi Distribuiti e Cloud Computing (A.A. 2025/26 - Tor Vergata).

---

## Requisiti e Installazione

Il sistema si basa su un'architettura a nodi che comunicano scambiandosi messaggi strutturati. Prima di avviare i componenti del progetto, è necessario configurare l'ambiente virtuale Python e installare le librerie richieste.

Invece di installare manualmente ogni singolo pacchetto, la procedura è automatizzata: l'installazione delle librerie pydantic (fondamentale per la validazione e lo scambio dei messaggi) e requests (utilizzata per gestire le chiamate di rete) avviene tramite il gestore di pacchetti pip.
u
### 1. Creazione e Attivazione dell'Ambiente Virtuale

È fortemente consigliato l'uso di un ambiente virtuale (venv) per isolare le dipendenze del progetto ed evitare conflitti con altre librerie globali sul computer.

Su Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

Su Mac / Linux (Terminal):
python3 -m venv venv
source venv/bin/activate

### 2. Installazione delle Dipendenze tramite requirements.txt

Una volta attivato l'ambiente virtuale, basterà eseguire il comando pip install puntando al file delle specifiche per scaricare e configurare tutto automaticamente:

pip install --upgrade pip
pip install -r requirements.txt


##Aggiunta anche di rpyc per la comunicazione

Prima di lanciare Swarm, dovrai fare:
* docker build -t tuo-utente-dockerhub/drf-worker:latest .
* docker push tuo-utente-dockerhub/drf-worker:latest

Poi : 
# Sul terminale del computer principale (Manager)
docker swarm init

Avviare il progetto: 
docker stack deploy -c docker-stack.yml mio-progetto-drf

Controllare lo stato dei servizi: 
docker service ls

Vedere su quali macchine fisiche stanno girando i singoli worker: 
docker stack ps mio-progetto-drf

Scalare dinamicamente i nodi:
docker service scale mio-progetto-drf_worker-federato=3

Leggere i log di un intero servizio distribuito: 
docker service logs mio-progetto-drf_orchestrator

Rimuove tutto il cluster: 
docker stack rm mio-progetto-drf

Dal momento che il progetto supporta differenti modalità operative, si è preferito aggiungere un file di configurazione in cui è possibile scegliere se lavorare in modalità federata oppure centralizzata, in locale oppure tramite aws learner lab. Per questo, a inizio progetto, si deve installare una libreria:  pip install python-dotenv


Inoltre, la classe Baseline rappresenta l'addestramento locale standard (non distribuito), nonnché quello realizzato su Colab, introdotto esclusivamente per poter effettuare un conftonto delle prestazioni con quello implementato in modo distribuito. 

Nel nostro progetto si è scelto di utilizzare l'SDK per python di AWS, ovvero Boto3 che è possibile installare attraverso il comando: pip install boto3 botocore. 

Inoltre, si procede con l'installazione di aws cli (seguendo le istruzioni della documentazione ufficiale seguendo il link riportato in descrizione a seconda del sistema operativo posseduto: https://docs.aws.amazon.com/it_it/cli/latest/userguide/getting-started-install.html)

Per scaricare il file da s3, bisogna installare le librerie: pip install fsspec s3fs

In realtà poi basta fare: pip install -r requirements.txt, poi per quanto riguarda aws instllazione di: pip install "botocore<1.43.0"


Aggiunto uno script per eseguire i ritardi di rete in locale. Possibilità di averli sia sul locale senza docker sia sul distribuito con docker: chmod +x run_local.sh. Dopodiché per eseguire il codice: ./run_local.sh delay. Inoltre, per poter avviare più terminali da bash, si è installato il seguente motore grafico: sudo dnf install gnome-terminal -y
Farlo anche per  run_test.sh, clean_local.sh

Prima di effettuare la build, vanno lanciati questi comandi: 
sudo chown -R $USER:$USER ./.local_storage
chmod -R 775 ./.local_storage


anche per la cartella ./.saved_models

prima vanno assegnate le variabili: 
export MY_UID=$(id -u)
export MY_GID=$(id -g)

Per lanciare: docker compose build, per poi eseguire docker compose up --scale worker=3 (da 1 a 7); 
Lanciare i comandi dei worker separatamente. 

Comando docker utile per eseguire in background: docker compose up --build --force-recreate -d

Invece, per quanto riguarda le cartelle Docker, ogni volta che si aggiungono: 
sudo chown -R $(id -u):$(id -g) workers_cache .local_storage saved_models
sudo chmod -R 777 workers_cache .local_storage saved_models

export MY_UID=$(id -u)
export MY_GID=$(id -g)
docker compose up

docker compose exec orchestrator python -m src.client.main

Per distruggere i processi attualmente attivi:  pkill -f "Distributed_RandomForest"

Per le credenziali aws lanciare ogni volta aws_creds.sh

esegui: export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)


Per uccidere i processi rimasti attivi: pkill -9 -f "Distributed_RandomForest"

Pacchetto da installare per i test su aws: curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" e in seguiti comando sudo (chiedere ad ia) -o; per verificare: "session-manager-plugin.deb"

Questo su Fedora: sudo dnf install -y ./session-manager-plugin.rpm