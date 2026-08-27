# Distributed_RandomForest

Sistema distribuito per il **training** e l'**inferenza** di modelli Random Forest, sviluppato per il progetto congiunto dei corsi di **Machine Learning** e **Sistemi Distribuiti e Cloud Computing** (A.A. 2025/26 — Università degli Studi di Roma Tor Vergata).

Il sistema segue un'architettura **master-worker**: un *orchestrator* centrale distribuisce l'addestramento dei singoli alberi della foresta su più nodi *worker*, in due modalità:

- **Centralizzata**: il dataset è caricato su uno storage condiviso (S3) e i worker addestrano porzioni della foresta sui medesimi dati.
- **Federata**: il dataset è pre-partizionato e distribuito sui nodi; ogni worker addestra localmente sui propri dati senza mai trasferire i dati grezzi al coordinatore, che si limita ad aggregare i modelli.

Sono supportati due ambienti di esecuzione, alternativi o combinabili:

| Ambiente | Come si avvia | Quando usarlo |
|---|---|---|
| **Locale / Docker Compose** | `docker compose up` | Sviluppo, test rapidi, simulazione di rete con `tc netem` |
| **AWS (ECS Fargate)** | `terraform apply` | Esecuzione "reale" su infrastruttura cloud, per gli esperimenti di scalabilità richiesti dal progetto |

---

## Indice

1. [Struttura del repository](#struttura-del-repository)
2. [Prerequisiti](#prerequisiti)
3. [Esecuzione in locale (Docker Compose)](#esecuzione-in-locale-docker-compose)
4. [Esecuzione su AWS (Terraform + ECS Fargate)](#esecuzione-su-aws-terraform--ecs-fargate)
5. [Modalità di training: centralizzata vs federata](#modalità-di-training-centralizzata-vs-federata)
6. [Simulazione/misura della latenza di rete](#simulazione-misura-della-latenza-di-rete)
7. [Test di sistema (performance, scalabilità, fault tolerance)](#test-di-sistema-performance-scalabilità-fault-tolerance)
8. [Pulizia / teardown](#pulizia--teardown)
9. [Limitazioni note](#limitazioni-note)

---

## Struttura del repository

```
.
├── src/
│   ├── client/            # entry point utente (sottomissione job, inferenza)
│   ├── master/orchestrator/  # coordinatore centrale (distribuzione, aggregazione, stato)
│   ├── worker/             # nodo di calcolo (addestramento locale dei singoli alberi)
│   ├── baseline/           # addestramento locale non distribuito, usato come riferimento
│   ├── shared/config.py    # caricamento configurazione da .env
│   └── testing/            # test engine di sistema (scenari 1-8) e generazione grafici
├── terraform/               # infrastruttura AWS as-code (ECR, S3, DynamoDB, SQS, ECS Fargate, API Gateway)
├── script_local/            # script per l'esecuzione locale
│   ├── run_local.sh         # avvio bare-metal senza Docker, multi-terminale
│   ├── run_test.sh          # avvio Docker Compose + test engine
│   └── clean_local.sh       # pulizia storage/modelli/cache locali
├── script_aws/               # script per il deploy/gestione manuale su AWS
│   ├── deploy.sh             # deploy via AWS CLI (alternativo a Terraform)
│   ├── run_aws.sh            # avvio client contro l'infrastruttura AWS
│   └── run_test_engine_ecs.sh   # test engine interattivo via ECS Exec
├── dataset_cache/            # cache locale dei dataset (CICIDS reale + sintetico)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── worker_supervisor.py       # restart-on-failure automatico dei worker (locale bare-metal e test engine)
├── upload_dataset.sh          # upload multipart con retry verso S3
└── aws_creds.sh               # helper per impostare le credenziali AWS Academy Learner Lab
```

> Se la struttura reale del tuo repository differisce da questa (nomi cartelle, script mancanti/aggiunti), aggiorna questa sezione prima di pubblicarlo: qui è ricostruita da quanto documentato nel progetto, non da un'ispezione diretta dei file.

---

## Prerequisiti

- **Python** 3.10+ e `venv`
- **Docker** e **Docker Compose** (per il flusso locale)
- **Terraform** >= 1.5 (per il flusso AWS)
- **AWS CLI v2**, configurato con le credenziali del tuo account (vedi sotto se usi un AWS Academy Learner Lab)
- Su Linux, i comandi di simulazione rete richiedono il pacchetto `iproute2` (fornisce `tc`)
- **Solo per l'esecuzione bare-metal senza Docker** (`run_local.sh`): un emulatore di terminale grafico, es. `gnome-terminal` (`sudo dnf install gnome-terminal` su Fedora) o `kgx`
- **Solo per entrare interattivamente nel test engine su AWS** (`run_test_engine_ecs.sh`, via `aws ecs execute-command`): il [Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) per l'AWS CLI (`.deb`/`.rpm` a seconda della distro)

> **Nota:** su alcuni sistemi `pip install -r requirements.txt` può installare una versione di `botocore` incompatibile con l'AWS CLI già presente. Se `aws` inizia a dare errori dopo l'installazione, forza una versione compatibile con `pip install "botocore<1.43.0"`.

---

## Esecuzione in locale (Docker Compose)

### 1. Clona il repository

```bash
git clone <URL_DEL_REPOSITORY>
cd Distributed_RandomForest
```

### 2. Crea e attiva un ambiente virtuale (opzionale ma consigliato se vuoi lanciare script Python fuori da Docker, es. `upload_dataset.sh` o gli script in `scripts/`)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configura il file `.env`

Crea un file `.env` nella root del progetto (non è versionato: contiene configurazione locale). Valori minimi per l'esecuzione locale:

```bash
ENV_MODE=local
TRAINING_MODE=federated        # oppure: centralized
NUM_WORKERS=3                  # da 1 a 7
```

`src/shared/config.py` legge queste variabili tramite `python-dotenv`; `TRAINING_MODE` deve essere esattamente `centralized` o `federated`, altrimenti il sistema si rifiuta di partire.

### 4. Prepara i permessi delle cartelle dati locali

Prima della prima build, assicurati che Docker possa scrivere nelle cartelle di storage locale:

```bash
mkdir -p .local_storage saved_models workers_cache
sudo chown -R $(id -u):$(id -g) .local_storage saved_models workers_cache
chmod -R 775 .local_storage saved_models workers_cache

export MY_UID=$(id -u)
export MY_GID=$(id -g)
```

### 5. Build e avvio

```bash
docker compose build
docker compose up --scale worker=3      # numero di worker a scelta, da 1 a 7
```

Per eseguirlo in background:

```bash
docker compose up --build --force-recreate -d
```

### 6. Avvia un job di training/inferenza dal client

In un altro terminale, con lo stack già in esecuzione:

```bash
docker compose exec orchestrator python -m src.client.main
```

Segui il prompt interattivo per scegliere dataset, iperparametri e modalità (vedi [sezione dedicata](#modalità-di-training-centralizzata-vs-federata)).

### 7. Fermare tutto

```bash
docker compose down
```

Per terminare eventuali processi rimasti attivi fuori da Compose (es. worker locali avviati senza Docker durante i test):

```bash
pkill -f "Distributed_RandomForest"     # oppure pkill -9 -f "Distributed_RandomForest" se non risponde
```

### Alternativa: esecuzione bare-metal senza Docker

`script_local/run_local.sh` avvia l'intero cluster **senza container**, direttamente sulla macchina host, aprendo un terminale grafico separato per ogni orchestrator e ogni worker (utile per osservare i log di ciascun nodo isolatamente, o per testare il failover multi-orchestratore). Richiede il `.env` già configurato e `worker_supervisor.py` nella root del progetto:

```bash
chmod +x script_local/run_local.sh
./script_local/run_local.sh puro     # avvio senza ritardo di rete
./script_local/run_local.sh delay    # avvio con 50ms di latenza artificiale su localhost (tc netem su 'lo')
```

Lo script chiede quanti orchestratori avviare (1 o 2), legge `NUM_WORKERS` dal `.env`, e infine apre il client interattivo nel terminale corrente. Con `delay`, il ritardo su `lo` viene applicato con `sudo tc` e rimosso automaticamente all'uscita (anche con Ctrl+C).

---

## Esecuzione su AWS (Terraform + ECS Fargate)

Questo flusso crea da zero l'infrastruttura AWS (ECR, S3, DynamoDB, SQS, ECS Fargate con Service per orchestrator/worker, API Gateway) con un singolo `terraform apply`, pensato per un account **AWS Academy Learner Lab**.

### 1. Credenziali AWS

Le credenziali del Learner Lab sono temporanee e scadono ogni ~4 ore. Recuperale dalla scheda "AWS Details" del Lab e impostale come variabili d'ambiente prima di ogni `terraform apply`:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```

Se scadono a metà `apply`, basta riesportarle e rilanciare: Terraform riprende dal proprio state senza ricreare risorse già esistenti. In alternativa, lo script `aws_creds.sh` genera/aggiorna `~/.aws/credentials`.

### 2. Configurazione

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# modifica terraform.tfvars: training_mode ("centralized"/"federated"), num_workers, ecc.
```

### 3. Deploy

```bash
terraform init
terraform plan     # opzionale, mostra cosa verrà creato
terraform apply
```

Il primo `apply` è più lento perché builda e pusha automaticamente l'immagine Docker su ECR (provisioner `null_resource.docker_build_push`): serve quindi **Docker installato e in esecuzione anche sulla macchina da cui lanci Terraform**. Al termine, l'output `next_steps` riporta i valori da copiare nel `.env` locale (bucket S3, endpoint API Gateway, regione, ecc.).

### 4. Upload del dataset su S3

```bash
./upload_dataset.sh    # multipart upload con retry automatico
```

Se il training è in modalità **federata**, prima di sottomettere un job va eseguito il provisioning degli shard per-nodo:

```bash
python -m scripts.provision_federated_shards --num-workers <N>
```

### 5. Esecuzione di un job contro l'infrastruttura AWS

Aggiorna il tuo `.env` locale con i valori d'output di Terraform (`ENV_MODE=aws`, `TRAINING_MODE`, `DATASETS_BUCKET_NAME`, `AWS_DEFAULT_REGION`, `NUM_WORKERS`, `API_GATEWAY_URL`), poi:

```bash
./run_aws.sh
```

Lo script attende che i Service ECS (worker + orchestrator) siano stabili prima di procedere, per non sottomettere job mentre l'infrastruttura sta ancora avviandosi.

### 6. Fermare/distruggere

Per scalare a zero senza distruggere l'infrastruttura (utile per pause tra sessioni di test):

```bash
aws ecs update-service --cluster forest-cluster --service orchestrator-service --desired-count 0 --region <REGION>
# ripeti per ciascun worker-service
```

Per distruggere tutto (ECS, ECR con l'immagine, S3 con contenuto, DynamoDB, SQS, Security Group) a fine sessione di valutazione:

```bash
cd terraform
terraform destroy
```

### Test engine interattivo su AWS

Per entrare nel menu interattivo del test engine (gli stessi scenari 1-8 del flusso locale) direttamente dentro la VPC, con i worker raggiungibili sul loro IP privato senza esporre le porte RPC su internet:

```bash
./script_aws/run_test_engine_ecs.sh
```

Lo script avvia un task ECS Fargate *one-off* (non un Service persistente) con `--enable-execute-command`, attende che l'agente ECS Exec sia pronto e apre una sessione interattiva equivalente al prompt locale. Richiede il [Session Manager Plugin](#prerequisiti) installato e, a fine sessione, chiede se fermare subito il task (consigliato, per non lasciare costi Fargate attivi inutilmente).

### Nota — script di deploy manuale alternativo

In `script_aws/` è presente anche `deploy.sh`, uno script bash equivalente che crea le stesse risorse via AWS CLI invece che via Terraform (utile per debug puntuale). Verifica le credenziali in `~/.aws/credentials` (generate con `bash aws_creds.sh`) e legge `TRAINING_MODE`/bucket dal `.env`. Il percorso consigliato per l'uso standard resta comunque Terraform.

---

## Modalità di training: centralizzata vs federata

Impostata tramite `TRAINING_MODE` nel `.env` (o `training_mode` in `terraform.tfvars` per AWS):

- **`centralized`**: dataset unico su S3 (o storage locale), il coordinatore distribuisce la costruzione dei singoli alberi tra i worker, che leggono tutti gli stessi dati.
- **`federated`**: il dataset è pre-partizionato (uno shard per nodo, generato con `provision_federated_shards.py` in ambiente AWS). Ogni worker addestra localmente sui propri dati e restituisce solo gli alberi addestrati, mai i dati grezzi.

La classe `Baseline` (in `src/baseline/`) rappresenta l'addestramento locale non distribuito (anche su Colab), usato esclusivamente come termine di paragone per la valutazione delle prestazioni richiesta dal progetto.

---

## Simulazione/misura della latenza di rete

Il progetto richiede di valutare l'impatto della latenza di rete tra i nodi, usando `tc`/`iproute2` con la capability Linux `CAP_NET_ADMIN` — punto discusso esplicitamente col docente durante il ricevimento sui Sistemi Distribuiti.

Il comportamento cambia in base all'ambiente:

- **Locale/Docker**: viene iniettato un ritardo artificiale reale con `tc netem` su un'interfaccia del container worker (altrimenti la latenza RPC su rete bridge Docker sarebbe pressoché nulla e non ci sarebbe nulla da misurare). La capability è già abilitata nel `docker-compose.yml` (`cap_add: NET_ADMIN`), quindi i comandi `tc` funzionano senza `sudo` dentro i container. Se lanci lo scenario di rete **fuori** da Docker (bare metal), serve invece una regola `NOPASSWD` in `/etc/sudoers` per `tc`, oppure lanciare l'intero engine con `sudo`; in assenza di permessi lo scenario prosegue comunque ma senza applicare un delay reale (stato `SKIPPED_NO_TC_PERMISSIONS`).
- **AWS/ECS Fargate**: `CAP_NET_ADMIN` **non è disponibile** nei task Fargate, e l'account AWS Academy Learner Lab usato per questo progetto non ha accesso ad AWS Fault Injection Simulator (verificato: `aws fis list-experiment-templates` → `AccessDeniedException`). Di conseguenza su AWS **non viene iniettato alcun ritardo artificiale**: lo scenario diventa invece una *misura* della latenza RPC reale tra i task (leader↔worker, stessa VPC, ENI separate), su più probe consecutivi. Questo valore **non è direttamente comparabile** al delay artificiale impostato in locale — vanno presentati nella relazione come due esperimenti distinti, non come lo stesso esperimento su due ambienti.

---

## Test di sistema (performance, scalabilità, fault tolerance)

Il modulo `src/testing/engine.py` espone un menu di scenari, eseguibile sia in locale (via `docker compose run --rm test-engine`, orchestrato da `script_local/run_test.sh`) sia su AWS (`script_aws/run_test_engine_ecs.sh`):

1. Performance e metriche
2. Scalabilità (al crescere del numero di nodi)
3. Simulazione di rete (vedi sezione precedente)
4. Guasto improvviso del worker (durante addestramento)
5. Guasto improvviso del worker (durante inferenza)
6. Failover dell'orchestratore (durante addestramento)
7. Failover dell'orchestratore (durante inferenza)
8. Generazione grafici a partire dai report salvati

Per l'esecuzione non interattiva (es. task ECS one-off senza terminale collegato), imposta la variabile d'ambiente `SCENARIO` con il numero desiderato o `all` invece di rispondere al prompt.

Avvio rapido in locale:

```bash
./script_local/run_test.sh
```

Lo script legge `NUM_WORKERS` dal `.env`, avvia lo stack con 2 istanze di orchestrator (per testare il failover) e il numero di worker configurato, esegue il test engine e poi ferma tutto.

---

## Pulizia / teardown

**Locale** — svuota storage, modelli salvati, cache worker e report di test (preservando la configurazione base):

```bash
./script_local/clean_local.sh
```

**AWS** — vedi [sezione 6 del flusso AWS](#6-fermaredistruggere) (`terraform destroy`).

---

## Limitazioni note

- Su AWS/Fargate la simulazione di rete non inietta un delay artificiale ma misura la latenza reale (vedi sopra): non confrontare direttamente i due esperimenti come se fossero equivalenti.
- L'ambiente Terraform assume un account **AWS Academy Learner Lab**: riusa il ruolo IAM `LabRole` già presente e la VPC di default. Fuori da un Learner Lab, `LabRole` non esiste e va sostituito con un ruolo IAM equivalente creato ad-hoc.
- Le credenziali AWS Academy scadono ogni ~4 ore: se un `terraform apply` o uno script si interrompe con errori di autenticazione, è quasi sempre questo il motivo.

---

## Autori

Progetto realizzato per i corsi di Machine Learning e Sistemi Distribuiti e Cloud Computing, A.A. 2025/26 — Tor Vergata.
Docenti: Valeria Cardellini, Gabriele Russo Russo.