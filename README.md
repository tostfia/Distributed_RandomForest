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