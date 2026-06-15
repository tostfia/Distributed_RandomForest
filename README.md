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