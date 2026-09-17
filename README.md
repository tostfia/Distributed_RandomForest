# Distributed_RandomForest

Sistema distribuito per il **training** e l'**inferenza** di modelli Random Forest, realizzato per il progetto congiunto dei corsi di **Machine Learning** e **Sistemi Distribuiti e Cloud Computing** (A.A. 2025/26 — Università degli Studi di Roma Tor Vergata), secondo la traccia ufficiale *"Progetto congiunto ML+SDCC 1: Training e Inferenza Distribuiti per Modelli Random Forest"*.

Il sistema segue l'architettura **master-worker**: un *orchestrator* centrale riceve dal client il dataset (URL a uno storage S3) e gli iperparametri del modello, distribuisce l'addestramento dei singoli alberi su più nodi *worker* e restituisce, a fine training, un **identificativo univoco del modello**. Le richieste di inferenza, identificate da quel model ID, vengono servite sfruttando l'infrastruttura distribuita e aggregando i risultati prodotti dai worker coinvolti. Il sistema gestisce, inoltre, la **tolleranza ai guasti** dei nodi durante training e inferenza, recuperando i risultati intermedi già salvati invece di far ripartire da zero i task falliti, e permette il download del modello addestrato in formato **Pickle** standard scikit-learn, per un utilizzo in un ambiente di inferenza locale.

Due modalità di training, selezionabili con `TRAINING_MODE`:

- **Centralizzata**: il dataset è caricato su uno storage condiviso e i worker addestrano porzioni della foresta sui medesimi dati.
- **Federata** : il dataset è già pre-partizionato e distribuito sui nodi: ogni worker addestra localmente sui propri dati senza mai trasferire i dati grezzi al coordinatore, che si limita ad aggregare i parametri del modello finale.

Le prestazioni vengono valutate confrontando il sistema con una baseline locale non distribuita (accuratezza e tempo di esecuzione, sia in training sia in inferenza) su più task di predizione, uno dei quali su dati sintetici generati con scikit-learn.

Sono supportati tre ambienti di esecuzione, alternativi o combinabili:

| Ambiente | Come si avvia | Differenze |
|---|---|---|
| **Locale (bare-metal)** | `run_local.sh` | Ogni nodo (worker, orchestrator) gira come processo separato direttamente sull'host, in un proprio terminale, senza container. |
| **Docker Compose** | `run_docker.sh` | Ogni nodo gira in un container Docker, con limiti di CPU/RAM configurabili da `.env`. |
| **AWS** | `run_aws.sh` | I worker girano su ECS Fargate, l'orchestrator su istanze EC2, il tutto provisionato da Terraform. |

---

## Indice

1. [Struttura del repository](#struttura-del-repository)
2. [Prerequisiti](#prerequisiti)
3. [Esecuzione in locale (Docker Compose)](#esecuzione-in-locale-docker-compose)
4. [Esecuzione su AWS (Terraform)](#esecuzione-su-aws-terraform)
5. [Modalità di training: centralizzata vs federata](#modalità-di-training-centralizzata-vs-federata)
6. [Simulazione e misura della latenza di rete](#simulazione-e-misura-della-latenza-di-rete)
7. [Test di sistema (performance, scalabilità, fault tolerance)](#test-di-sistema-performance-scalabilità-fault-tolerance)
8. [Pulizia](#pulizia)


---

## Struttura del repository

```
.
├── src/
│   ├── client/              # entry point utente (sottomissione job, inferenza) — main.py
│   ├── orchestrator/        # coordinatore centrale — main.py, BaseOrchestrator.py, centralized.py, federated.py
│   ├── worker/               # nodo di calcolo — main.py, BaseWorker.py, centralizedWorker.py, federatedWorker.py
│   ├── dataset/              # layer DAO per storage dati/checkpoint/metriche (S3 o locale)
│   │   ├── dataset_dao.py            # accesso al dataset (S3/locale), caching, pyarrow
│   │   ├── dataset_dao_factory.py    # selezione DAO in base ad ENV_MODE
│   │   ├── checkpoint_dao.py         # persistenza checkpoint di training
│   │   └── metrics_dao.py            # persistenza metriche/report
│   ├── baseline/             # addestramento locale non distribuito, usato come riferimento — run_baseline.py e diagnostica
│   ├── shared/                # utilità condivise tra client/orchestrator/worker
│   │   ├── config.py                 # caricamento configurazione da .env
│   │   ├── factory.py                # factory generiche (worker/orchestrator per ambiente)
│   │   ├── binding/                  # ServiceRegistry e binding RPyC
│   │   ├── sharedmodels/             # modelli dati condivisi
│   │   ├── mock_aws/                 # mock locali dei servizi AWS (per esecuzione senza cloud)
│   │   └── utilities/                # loader dataset, splitter, task_storage, ecc.
│   └── testing/               # test engine di sistema
│       ├── engine.py                 # entry point, selezione scenario
│       ├── scenarios/                # implementazione dei singoli scenari (1-10)
│       ├── plot_generator.py         # scenario 10, grafici da report salvati
│       └── test_config.json          # configurazione degli scenari di test
├── terraform/               # infrastruttura AWS as-code (ECR, S3, DynamoDB, SQS, ECS Fargate, EC2/ASG orchestrator, API Gateway, EFS) — vedi terraform/README.md
├── lambda_source/           # sorgente della funzione Lambda usata da Terraform per il deploy (copia distinta da terraform/lambda/ e src/shared/mock_aws/lambda/)
├── script_local/            # script per l'esecuzione locale
│   ├── run_local.sh              # avvio bare-metal senza Docker, multi-terminale
│   ├── run_docker.sh             # avvio Docker Compose RACCOMANDATO: provisioning + rete + limiti CPU/RAM da .env
│   ├── run_test.sh               # avvio Docker Compose + test engine (per i test di sistema, sez. 7)
│   ├── provision_local_shards.py # provisioning offline degli shard federati su disco (gemello locale dello script AWS)
│   ├── clean_local.sh            # pulizia selettiva di storage/modelli/cache locali
│   └── preserve_baseline_boot.py # helper di clean_local.sh: preserva la config baseline attraverso il reset
├── script_aws/               # script operativi contro l'infrastruttura AWS già deployata da Terraform
│   ├── run_aws.sh                    # avvio client contro l'infrastruttura AWS
│   ├── run_test_engine.sh        # test engine su EC2 on demand
│   ├── provision_federated_shards.py # provisioning offline degli shard federati su S3
│   ├── teardown.sh                   # scala i Service/l'ASG a 0 e svuota DynamoDB/SQS/S3 (senza distruggere l'infrastruttura)
│   ├── check_left_over.sh            # controllo read-only di risorse AWS rimaste attive per errore
│   └──  aws_creds.sh               # helper per impostare le credenziali AWS Academy Learner Lab
├── outputs_baseline/         # manifesti/modelli prodotti da run_baseline.py: config_real.json, config_synthetic.json
│                             # (feature selection + iperparametri, fonte di verità condivisa col training distribuito)
├── dataset_cache/            # cache locale dei CSV grezzi del dataset reale (CICIDS)
├── synthetic/                # cache locale del dataset sintetico generato
├── saved_models/             # modelli distribuiti salvati (generati a runtime)
├── workers_cache/            # cache locale lato worker (generata a runtime)
├── test_reports/             # report dei test engine (local/, docker/ e aws/, generati a runtime)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── worker_supervisor.py       # restart-on-failure automatico dei worker (locale bare-metal e test engine)
└── upload_dataset.sh          # upload multipart con retry verso S3
 
```

> Le cartelle `outputs_baseline/`, `dataset_cache/`, `synthetic/`, `saved_models/`, `workers_cache/`, `test_reports/` sono in gran parte popolate a runtime (modelli, cache, report).

---

## Prerequisiti

- **Python** 3.10+ e `venv`
- **Docker** e **Docker Compose** (per il flusso locale)
- **Terraform** >= 1.5 (per il flusso AWS)
- **AWS CLI v2**, configurato con le credenziali del tuo account (vedi sotto se usi un AWS Academy Learner Lab)
- Su Linux, i comandi di simulazione rete richiedono il pacchetto `iproute2` (fornisce `tc`)
- **Solo per l'esecuzione bare-metal senza Docker** (`run_local.sh`): un emulatore di terminale grafico, es. `gnome-terminal` (`sudo dnf install gnome-terminal` su Fedora) o `kgx`

> **Nota:** su alcuni sistemi `pip install -r requirements.txt` può installare una versione di `botocore` incompatibile con l'AWS CLI già presente. Se `aws` inizia a dare errori dopo l'installazione, forza una versione compatibile con `pip install "botocore<1.43.0"`.

---

## Esecuzione in locale (Docker Compose)

### 1. Clona il repository

```bash
git clone <URL_DEL_REPOSITORY>
cd Distributed_RandomForest
```

### 2. Crea e attiva un ambiente virtuale (opzionale ma consigliato se vuoi lanciare script Python fuori da Docker, es. `upload_dataset.sh` o gli script in `script_aws/`)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configura il file `.env`

Il sistema viene configurato tramite il file `.env` presente nella root del progetto. È possibile crearlo partendo dal modello `.env.example`:

```bash
cp .env.example .env
```
**Modalità di esecuzione**

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **RUNNING_IN_DOCKER** | `true/false` | Indica se l'applicazione gira dentro un container Docker. La impostano già gli script (`run_docker.sh`, `run_test.sh`): non serve toccarla a mano, a meno di lanciare `docker compose up` manualmente. |
| **TRAINING_MODE** | `centralized/federated` | Seleziona la modalità di addestramento. In pratica va sempre impostata esplicitamente: i vari script che la leggono hanno fallback diversi tra loro se assente, con il rischio di far partire componenti in modalità incoerenti. |
| **ENV_MODE** | `local/aws` | Seleziona l'ambiente di esecuzione: `local` per Docker/host, `aws` per Fargate/EC2/S3: determina quale orchestrazione infrastrutturale usare. Va sempre impostata esplicitamente, per lo stesso motivo di `TRAINING_MODE`. |
| **DATASET_TYPE** | `real/synthetic` | Specifica se caricare il dataset reale (CICIDS) o generare un dataset sintetico. Se la ometti, gli script di provisioning ricadono su `real`, ma il menu interattivo del client te lo richiede comunque a ogni avvio, quindi impostarla qui serve solo per gli script non interattivi (provisioning, test engine). `real` richiede il dataset scaricato in locale (vedi passo 4 più sotto), `synthetic` no. |
| **SYNTHETIC_N_SAMPLES** | Numero intero | Numero di campioni generati se `DATASET_TYPE=synthetic`. Ignorata con `DATASET_TYPE=real`. |
| **CENTRALIZED_DATASET_MODE** | `shared/sharded` | Solo per `TRAINING_MODE=centralized` (ignorata in federated). Se la ometti, il sistema usa `shared`: ogni worker scarica l'intero dataset. Con `sharded`, invece, il dataset viene partizionato e ogni worker scarica solo una fetta: vedi [Modalità di training](#modalità-di-training-centralizzata-vs-federata) per il comportamento diverso tra reale e sintetico. |

**Dimensionamento del cluster (locale/Docker)**

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **NUM_WORKERS** | Numero intero | Quanti worker avviare in locale/Docker (in AWS federated: quanti indici/shard fissi crea Terraform, vedi `terraform/README.md`). Se la ometti, il default varia da uno script all'altro (2 in alcuni casi, 3 in altri): impostala sempre esplicitamente, altrimenti client e provisioning potrebbero ragionare su un numero di worker diverso. |
| **WORKER_CORES** | Numero intero (opzionale) | Override esplicito di quanti processi/thread paralleli usa UN worker per costruire gli alberi. **In Docker Compose è sempre impostata**: se la ometti nel `.env`, `docker-compose.yml` la valorizza comunque a `2` di suo (non lascia decidere al calcolo dinamico). Il calcolo dinamico (core disponibili ÷ worker attivi sulla stessa macchina) entra in gioco solo con `run_local.sh` (bare-metal), dove nessuno imposta questa variabile di default. Impostarla esplicitamente serve per esperimenti di strong scaling, dove ogni worker deve rappresentare una capacità di calcolo fissa e comparabile, indipendente da quanti altri worker girano in parallelo. **Non va confusa con `WORKER_CPUS`**, che è un limite Docker, non un parametro applicativo. |
| **WORKER_CPUS** / **ORCHESTRATOR_CPUS** | Numero (es. `1`, `0.5`) | Limite CPU Docker Compose per container worker/orchestrator (evita di saturare la macchina di sviluppo con `NUM_WORKERS` alto). Se le ometti, `docker-compose.yml` applica comunque un default proprio: `2.0` per il worker, `0.5` per l'orchestrator. |
| **WORKER_MEM_LIMIT** / **ORCHESTRATOR_MEM_LIMIT** | Es. `2048m` | Limite di memoria Docker Compose per container worker/orchestrator. Stesso discorso di `WORKER_CPUS`: se assenti, `docker-compose.yml` ricade sui suoi default (`2048m` worker, `1024m` orchestrator). |
| **IMAGE_NAME** | Stringa | Nome:tag dell'immagine Docker locale (es. `rf-worker-local:latest`), usato da `docker-compose.yml` per sapere quale immagine costruire/avviare. |
| **EC2_ID** | Stringa libera | Etichetta usata solo per comporre il nome interno dell'orchestratore (log, lock di leadership su DynamoDB) — non incide sulla logica applicativa. Se la ometti, il default nel codice è `Locale`; su AWS Terraform la imposta fissa a `EC2Orchestrator`. |
| **MY_UID** / **MY_GID** | Numero intero | UID/GID mappati dentro i container per i permessi delle cartelle di storage locale. I valori nel template sono solo un punto di partenza: se non li sostituisci con `$(id -u)`/`$(id -g)` del tuo utente (vedi passo 5 sotto) prima della build, rischi errori di permessi sui volumi montati. |

**Supervisor dei worker federati** (restart automatico in caso di crash)

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **FED_SUPERVISOR_MAX_RESTARTS** | Numero intero | Tentativi di restart automatico per worker federato caduto. `0` disabilita del tutto il restart automatico. |
| **FED_SUPERVISOR_BACKOFF_SECONDS** | Numero intero | Attesa, in secondi, prima del primo tentativo di restart. |
| **FED_SUPERVISOR_BACKOFF_MAX_SECONDS** | Numero intero | Tetto massimo dell'attesa tra tentativi successivi (il backoff cresce fino a questo valore, poi si ferma). |
| **FED_WORKER_WAIT_TIMEOUT_SECONDS** | Numero intero | Timeout di attesa per il rientro di un worker sostituito, usato dagli scenari di fault tolerance. Se la ometti, il default nel codice è 60s — troppo poco su AWS reale (un rimpiazzo Fargate misurato empiricamente ha richiesto 81s dal kill alla steady state). Il valore 120 nel template lascia margine sopra quell'osservazione. |

**Partizionamento federato** (solo `TRAINING_MODE=federated`, `DATASET_TYPE=real`)

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **PARTITION_STRATEGY** | `by_day/iid` | Strategia di partizionamento dello shard federato. Se la ometti, il default nel codice è `iid` (mescolamento casuale globale). `by_day` partiziona invece per giorno/file di origine — vedi [Modalità di training](#modalità-di-training-centralizzata-vs-federata). |
| **DAY_COLUMN** | Stringa (opzionale, solo con `PARTITION_STRATEGY=by_day`) | Nome della colonna da usare per partizionare per giorno. **Non obbligatoria**: se omessa, il sistema usa automaticamente la colonna generata dal loader (`_capture_day`) — `by_day` funziona senza configurarla. Impostarla serve solo per usare una colonna diversa già presente nel dataset. |

**Timeout RPC**

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **RPC_SYNC_TIMEOUT_SECONDS** | Numero intero | Timeout per le chiamate RPC sincrone di training. Se la ometti, il default nel codice è 1800s (30 minuti). |
| **RPC_INFERENCE_SYNC_TIMEOUT_SECONDS** | Numero intero | Timeout per le chiamate RPC sincrone di inferenza. Se la ometti, il default nel codice è 900s (15 minuti). |

**Sorgenti dati**

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **DATASET_LOCAL_PATH** | Path | Cartella di cache locale per i CSV grezzi del dataset reale. Se la ometti, il default nel codice è `./dataset_cache`. |
| **DEFAULT_DATASET_S3_URL** | URL S3 | URL S3 pubblico del dataset CICIDS2018 (sorgente esterna, non un bucket del progetto). **Puramente informativa**: nessuno script la legge davvero: il client ha lo stesso URL scritto direttamente nel codice come fallback per il dataset reale in locale. Impostarla o ometterla non cambia il comportamento del sistema; serve solo a chi legge il `.env` per sapere da dove viene il dataset. |

**Risorse AWS** (solo `ENV_MODE=aws` — valori specifici dell'account, non committare quelli reali)

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **DATASETS_BUCKET_NAME** | Stringa | Nome del bucket S3 dei dataset (valore d'output di `terraform apply`, vedi `terraform/README.md`). Senza questa variabile, gli script AWS (provisioning, upload, test engine) falliscono esplicitamente invece di usare un bucket di default. |
| **AWS_DEFAULT_REGION** | Es. `us-east-1` | Regione AWS del deploy. Se la ometti, gli script ricadono su `us-east-1`. |
| **API_GATEWAY_URL** | URL | Endpoint API Gateway esposto dal deploy Terraform — **cambia ad ogni ricreazione dello stack**, va aggiornato dopo ogni `apply`. Se lasci un valore vecchio, il client non fallisce in modo esplicito: parla semplicemente con un endpoint che non esiste più. |


### 4. Procurati il dataset reale (solo se `DATASET_TYPE=real`)

Se hai impostato `DATASET_TYPE=synthetic`, salta questo passo: ogni worker genera il proprio dataset sintetico al boot, nessun file va scaricato.

Per `DATASET_TYPE=real` (dataset **CICIDS2018**) il comportamento dipende da cosa stai per lanciare:

- **Training centralizzato tramite il client**: non serve fare nulla in anticipo. Il client passa direttamente l'URL del bucket pubblico S3 (`s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/`), e il loader lo scarica automaticamente con accesso anonimo (nessuna credenziale AWS richiesta) se non trova già i CSV in `dataset_cache/`. **Attenzione**: questo download non viene salvato automaticamente: se lasci `dataset_cache/` vuota, ogni run scarica di nuovo tutto da S3.
- **Modalità federata (provisioning locale) e baseline locale** (`run_baseline.py`): qui invece i CSV devono essere **già presenti** in `dataset_cache/` (o nel path indicato da `DATASET_LOCAL_PATH`): questi due script non hanno alcun fallback su S3 e falliscono con un errore esplicito se la cartella è vuota o assente.

In entrambi i casi, per evitare di riscaricare da S3 a ogni esecuzione, conviene popolare `dataset_cache/` una volta sola:

```bash
mkdir -p dataset_cache
aws s3 sync "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" ./dataset_cache --no-sign-request
```

`--no-sign-request` funziona perché è un bucket pubblico dell'AWS Open Data Registry: non servono credenziali AWS per questo download. Se il comando fallisce, verifica sulla pagina ufficiale del dataset ([registry.opendata.aws](https://registry.opendata.aws/cse-cic-ids2018/)) che il path non sia cambiato.

### 5. Prepara i permessi delle cartelle dati locali

Prima della prima build, assicurati che Docker possa scrivere nelle cartelle di storage locale:

```bash
mkdir -p .local_storage saved_models workers_cache
sudo chown -R $(id -u):$(id -g) .local_storage saved_models workers_cache
chmod -R 775 .local_storage saved_models workers_cache

export MY_UID=$(id -u)
export MY_GID=$(id -g)
```

### 6. Build e avvio

**Consigliato: `run_docker.sh` questo script:
- esegue automaticamente il **provisioning degli shard federati** se `TRAINING_MODE=federated` (senza, l'orchestrator si aspetta shard già presenti e non li genera più a runtime — vedi `script_local/provision_local_shards.py`);
- **gestisce il ritardo di rete per te**: `docker-compose.yml` applica `delay 0ms` di default (nessun ritardo) su ogni worker se la variabile `NET_SCENARIO` non è impostata, senza passare da questo script, parte pulito, senza latenza artificiale indesiderata. `run_docker.sh puro` imposta comunque esplicitamente `NET_SCENARIO="delay 0ms"` (stesso valore del default, ma dichiarato invece che implicito), `run_docker.sh delay` la imposta a `delay 50ms` per introdurre latenza voluta quando serve testarne l'impatto;
- applica i limiti di CPU/RAM da `.env` (`WORKER_CPUS`, `WORKER_MEM_LIMIT`, ecc.), utile per non saturare la macchina di sviluppo con `NUM_WORKERS` alto.

```bash
chmod +x script_local/run_docker.sh
./script_local/run_docker.sh puro     # avvio senza ritardo di rete
./script_local/run_docker.sh delay    # avvio con 50ms di latenza artificiale
```

Lo script chiede il numero di orchestratori (1-2), legge `NUM_WORKERS`/`TRAINING_MODE` dal `.env`, builda l'immagine se necessario e avvia il cluster in background; il client resta sull'host e parte automaticamente al termine.


### Alternativa: esecuzione bare-metal senza Docker

`script_local/run_local.sh` avvia l'intero cluster **senza container**, direttamente sulla macchina host, aprendo un terminale grafico separato per ogni orchestrator e ogni worker (utile per osservare i log di ciascun nodo isolatamente, o per testare il failover multi-orchestratore). Richiede il `.env` già configurato e `worker_supervisor.py` nella root del progetto:

```bash
chmod +x script_local/run_local.sh
./script_local/run_local.sh puro     # avvio senza ritardo di rete
./script_local/run_local.sh delay    # avvio con 50ms di latenza artificiale su localhost (tc netem su 'lo')
```

Lo script chiede quanti orchestratori avviare (1 o 2), legge `NUM_WORKERS` dal `.env`, e infine apre il client interattivo nel terminale corrente. Con `delay`, il ritardo su `lo` viene applicato con `sudo tc` e rimosso automaticamente all'uscita (anche con Ctrl+C).

---

## Esecuzione su AWS (Terraform)

Questo flusso crea da zero l'infrastruttura AWS (ECR, S3, DynamoDB, SQS, ECS Fargate con Service per worker, EC2/Auto Scaling Group per l'orchestrator, API Gateway) con un singolo `terraform apply`, pensato per un account **AWS Academy Learner Lab**. Per i dettagli completi (incluse le restrizioni SCP del Learner Lab e come aggirarle) vedi **[`terraform/README.md`](terraform/README.md)**; qui il riassunto operativo.

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

Se il training è in modalità **federata**, prima di sottomettere un job va eseguito il provisioning degli shard per-nodo. Il tuo `.env` deve avere `DATASETS_BUCKET_NAME` impostato (lo script fallisce esplicitamente se manca, invece di usare un bucket di default):

```bash
python -m script_aws.provision_federated_shards --num-workers <N>
```

### 5. Esecuzione di un job contro l'infrastruttura AWS

Aggiorna il tuo `.env` locale con i valori d'output di Terraform (`ENV_MODE=aws`, `TRAINING_MODE`, `DATASETS_BUCKET_NAME`, `AWS_DEFAULT_REGION`, `NUM_WORKERS`, `API_GATEWAY_URL`), poi:

```bash
./script_aws/run_aws.sh
```

Lo script attende che i Service ECS (worker + orchestrator) siano stabili prima di procedere, per non sottomettere job mentre l'infrastruttura sta ancora avviandosi.

### 6. Fermare/distruggere

Per scalare a zero senza distruggere l'infrastruttura, il modo più completo è `script_aws/teardown.sh` — scala i worker e l'Auto Scaling Group dell'orchestrator a 0 **e** svuota le tabelle DynamoDB e le code SQS (stato applicativo pulito, schema e infrastruttura intatti):

```bash
./script_aws/teardown.sh
```

In alternativa, per fermare solo l'esecuzione senza toccare lo stato applicativo:

```bash
aws autoscaling update-auto-scaling-group --auto-scaling-group-name orchestrator-asg \
  --min-size 0 --max-size 0 --desired-capacity 0 --region <REGION>
aws ecs update-service --cluster forest-cluster --service worker-service --desired-count 0 --region <REGION>
# in modalità federated, ripeti l'ultimo comando per ciascun worker-service-<N>
```

Prima di chiudere una sessione di lavoro, `script_aws/check_left_over.sh` verifica che non sia rimasto nulla attivo che continui a fatturare:

```bash
./script_aws/check_left_over.sh
```

Per distruggere tutto a fine sessione di valutazione:

```bash
cd terraform
terraform destroy
```

> Il bucket S3 e i log group CloudWatch, creati manualmente per aggirare le restrizioni SCP del Learner Lab (vedi `terraform/README.md`), **non** vengono rimossi da `terraform destroy` — richiedono pulizia manuale separata se vuoi eliminarli del tutto.

### Test engine su AWS (istanza EC2 usa e getta)

Per eseguire uno degli scenari di test (1-10 o `all`) direttamente dentro la VPC, con i worker raggiungibili sul loro IP privato senza esporre le porte RPC su Internet:

```bash
./script_aws/run_test_engine.sh <scenario>      # es. ./script_aws/run_test_engine.sh 2
./script_aws/run_test_engine.sh                 # chiede lo scenario a terminale prima di lanciare l'istanza
```
Lo script avvia un'istanza EC2 usa-e-getta ed esegue il container Docker con lo scenario passato via variabile d'ambiente `SCENARIO`. L'istanza prosegue in background anche se chiudi il terminale e si autodistrugge automaticamente (`shutdown -h now`) al termine del test; i log finiscono su CloudWatch (`/ec2/rf-test-engine`) e il report finale viene caricato su `s3://<bucket>/test_reports/aws/`.

```bash
aws logs tail /ec2/rf-test-engine --follow --region <REGION>   # segui i log in tempo reale
```
---

## Modalità di training: centralizzata vs federata

Impostata tramite `TRAINING_MODE` nel `.env` (o `training_mode` in `terraform.tfvars` per AWS):

- **`centralized`**: dataset unico su S3 (o storage locale), il coordinatore distribuisce la costruzione dei singoli alberi tra i worker. Due sotto-modalità, selezionate da `CENTRALIZED_DATASET_MODE` nel `.env`:
  - **`shared`** (default): ogni worker scarica l'**intero** dataset: comportamento storico, identico per qualunque `DATASET_TYPE`.
  - **`sharded`**: il dataset viene partizionato e ogni worker scarica solo una fetta, per ridurre il traffico di rete per worker. Il criterio di partizionamento **dipende dal tipo di dataset**, non è lo stesso in entrambi i casi:
    - **Sintetico**: numero di shard = numero di worker rilevati al momento (dinamico, un worker = uno shard).
    - **Reale**: numero di shard **fisso** (indipendente dal numero di worker, per permettere il riuso degli stessi file tra round di scaling diversi), e ogni worker può ricevere **più shard**, che unisce localmente prima del training.
- **`federated`**: il dataset è pre-partizionato (uno shard per nodo, generato con `provision_federated_shards.py` in ambiente AWS). Ogni worker addestra localmente sui propri dati e restituisce solo gli alberi addestrati, mai i dati grezzi. Dovranno essere impostate le seguenti variabili nel file `.env`:

| Variabile | Valori ammessi | Descrizione |
|---|---|---|
| **PARTITION_STRATEGY** | `by_day/iid` | Strategia di partizionamento dello shard federato. |


La classe `Baseline` (in `src/baseline/`) rappresenta l'addestramento locale non distribuito, usato esclusivamente come termine di paragone per la valutazione delle prestazioni richiesta dal progetto.

---

## Simulazione e misura della latenza di rete

Dal momento che in locale/Docker la latenza reale tra i container è pressoché nulla (rete bridge), si è introdotto un ritardo artificiale con `tc`/`iproute2` (capability Linux `CAP_NET_ADMIN`). Su AWS, dove questa strada non è percorribile (vedi sotto), si è adottato un approccio differente.

Il comportamento cambia in base all'ambiente:

- **Locale/Docker**: viene iniettato un ritardo artificiale reale con `tc netem` su un'interfaccia del container worker (altrimenti la latenza RPC su rete bridge Docker sarebbe pressoché nulla). La capability è già abilitata nel `docker-compose.yml` (`cap_add: NET_ADMIN`), quindi i comandi `tc` funzionano senza `sudo` dentro i container. Se si lancia lo scenario di rete **fuori** da Docker (bare metal), serve invece una regola `NOPASSWD` in `/etc/sudoers` per `tc`, oppure si deve lanciare l'intero engine con `sudo`; in assenza di permessi lo scenario prosegue comunque ma senza applicare un delay reale (stato `SKIPPED_NO_TC_PERMISSIONS`).
- **AWS/ECS Fargate**: `CAP_NET_ADMIN` **non è disponibile** nei task Fargate e l'account AWS Academy Learner Lab usato per questo progetto non ha accesso ad AWS Fault Injection Simulator. Di conseguenza su AWS **non viene iniettato alcun ritardo artificiale**: lo scenario diventa invece una *misura* della latenza RPC reale tra i task (leader↔worker, stessa VPC, ENI separate), su più probe consecutivi.

---

## Test di sistema (performance, scalabilità, fault tolerance)

La validazione e la verifica dell'architettura distribuita sono affidate ad un **Test Engine** automatizzato (`src/testing/engine.py`). L'engine permette di eseguire una suite completa di scenari sia in **ambiente locale** (tramite gli script dedicati) sia su **AWS** (`./script_aws/run_test_engine.sh`), raccogliendo metriche e salvando i report finali.

### Locale/Docker

```bash
chmod +x script_local/run_test.sh
./script_local/run_test.sh
```

Lo script legge `NUM_WORKERS`/`TRAINING_MODE` dal `.env` (provisionando gli shard federati automaticamente se necessario, come `run_docker.sh`), poi avvia il cluster e apre il container `test-engine` in modalità **interattiva**: da lì si sceglie lo scenario da eseguire (menu identico a quello mostrato sotto). Al termine il cluster viene fermato automaticamente. I report finiscono in `test_reports/docker/`.

I test disponibili coprono le seguenti aree operative:
1. Performance e metriche
2. Scalabilità (al crescere del numero di nodi)
3. Simulazione di rete (vedi sezione precedente)
4. Guasto improvviso del worker (durante addestramento)
5. Guasto improvviso del worker (durante inferenza)
6. Failover dell'orchestratore (durante addestramento)
7. Failover dell'orchestratore (durante inferenza)
8. Elezione del leader sotto concorrenza (safety)
9. Sostituzione ASG dell'Orchestratore (solo AWS)
10. Generazione grafici a partire dai report salvati

> **Nota sullo scenario 10**: a differenza degli altri, non esegue training/inferenza: legge i report JSON già salvati in `test_reports/` (priorità `aws > docker > local`) per produrre i grafici della relazione. Selezionando `all`, viene eseguito automaticamente **per ultimo**, dopo tutti gli altri scenari (inclusa la sostituzione ASG). Se lanci gli scenari singolarmente uno alla volta, esegui prima quelli che ti interessano e lancia il `10` solo alla fine. Se nella stessa cartella `test_reports/<ambiente>/` convivono report con configurazioni diverse (numero di alberi, dimensione dataset), sono esperimenti non confrontabili: il generatore li tiene separati e lo segnala a schermo.

### AWS

Vedi [Test engine su AWS](#test-engine-su-aws-istanza-ec2-usa-e-getta) nella sezione precedente: stesso menu di scenari, eseguito su un'istanza EC2 usa-e-getta con `./script_aws/run_test_engine.sh <scenario>`.


## Pulizia

**Locale** — pulizia **selettiva**, non totale: svuota `.local_storage/`, `saved_models/`, `workers_cache/` e `test_reports/local/`, ma preserva esplicitamente due cose attraverso il reset:

```bash
./script_local/clean_local.sh
```

- **`.local_storage/metrics/`** non viene toccata da non perdere lo storico delle metriche tra una sessione di test e l'altra.
- **La sezione `baseline_boot`** di `.local_storage/config.json` (dataset_type, tree_type) sopravvive al reset tramite `preserve_baseline_boot.py`, che la estrae prima della pulizia e la reintegra subito dopo; tutto il resto del config (in particolare `last_training_request` e lo storico delle richieste) viene invece azzerato come da comportamento previsto.

Se invece serve un reset totale, anche di queste due eccezioni, va fatto a mano (es. cancellando direttamente `.local_storage/metrics/` o l'intero `.local_storage/config.json`).

**AWS** — due livelli, dal meno al più distruttivo:

1. `./script_aws/teardown.sh` — scala i worker e l'Auto Scaling Group dell'orchestrator a 0 e svuota lo stato applicativo (DynamoDB, SQS, artefatti S3 temporanei), lasciando intatte task definition/cluster/ECR/ASG per un riavvio rapido. Supporta `--purge-shards`, `--purge-legacy-mode`, `--purge-models` (vedi commenti in testa allo script).
2. `terraform destroy` — rimuove tutta l'infrastruttura (vedi [sezione 6 del flusso AWS](#6-fermaredistruggere)).

In entrambi i casi, prima di chiudere una sessione conviene lanciare `./script_aws/check_left_over.sh` per un controllo finale di eventuali risorse rimaste attive per errore.

---

## Autori

Progetto realizzato per i corsi di Machine Learning e Sistemi Distribuiti e Cloud Computing, A.A. 2025/26 — Università degli studi di Roma "Tor Vergata".
Docenti: Prof.ssa Valeria Cardellini, Prof. Gabriele Russo Russo.