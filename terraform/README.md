# Infrastruttura AWS via Terraform — Distributed_RandomForest

Questo modulo Terraform crea **da zero** tutta l'infrastruttura AWS necessaria
al sistema (ECR, S3, DynamoDB, SQS, ECS Fargate + Service orchestrator/worker),
buildando e pushando anche l'immagine Docker dell'applicazione. Pensato per
essere eseguito con un **singolo `terraform apply`** in un account
**AWS Academy Learner Lab**.

> ⚠️ Un Learner Lab impone alcune restrizioni particolari (SCP) che richiedono
> pochi passaggi manuali una tantum prima del primo deploy. Sono descritti
> nella sezione [Setup manuale richiesto](#3-setup-manuale-richiesto-solo-learner-lab)
> — **non saltarla**, altrimenti il primo `terraform apply` fallisce.

## Prerequisiti

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- Docker installato e in esecuzione (il modulo builda e pusha l'immagine
  automaticamente)
- AWS CLI configurato con le credenziali del Learner Lab (vedi sotto)
- Essere nella **root del progetto** con questa cartella `terraform/` al suo
  interno, allo stesso livello del `Dockerfile`

## 1. Credenziali AWS (Learner Lab)

Le credenziali del Learner Lab sono **temporanee** e scadono ogni ~4 ore.
Prendile dalla scheda "AWS Details" del Lab (pulsante "Show" su
AWS CLI) e impostale come variabili d'ambiente PRIMA di lanciare Terraform:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```

Se durante `terraform apply` le credenziali scadono, aggiornale con gli
stessi comandi e rilancia `terraform apply`: Terraform riprende da dove si
era fermato grazie al proprio state, senza ricreare le risorse già create.

**Non condividere mai queste credenziali** (nemmeno temporaneamente, es. in
chat, ticket, o commit): finché sono valide, chiunque le legga può operare
sul tuo account Lab. Se sospetti che siano state esposte, chiudi la sessione
(End Lab) e riaprila (Start Lab) per invalidarle.

## 2. Configurazione

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# modifica terraform.tfvars se necessario (es. training_mode, num_workers)
```

## 3. Setup manuale richiesto (solo Learner Lab)

Gli account AWS Academy Learner Lab applicano delle Service Control Policy
(SCP) centralizzate più restrittive del normale. Alcune chiamate che
Terraform farebbe automaticamente vengono negate, e vanno quindi anticipate
a mano **una sola volta** prima del primo `apply`. Nella nostra esperienza
diretta abbiamo isolato tre restrizioni:

### 3.1 Bucket S3 dei dataset

Durante la pianificazione dei bucket S3, il provider Terraform interroga
l'API `GetBucketObjectLockConfiguration`. Nei Learner Lab questa chiamata è
esplicitamente negata dalla SCP con `AccessDenied`. Per aggirarlo, crea il
bucket manualmente prima di lanciare Terraform (che poi lo userà come
risorsa/data source già esistente):

```bash
aws s3api create-bucket \
  --bucket rf-distributed-datasets-<ACCOUNT_ID>-us-east-1 \
  --region us-east-1
```

Sostituisci `<ACCOUNT_ID>` con il tuo Account ID (visibile in alto a destra
nella Console, o con `aws sts get-caller-identity --query Account --output text`).
Il nome deve corrispondere **esattamente** a quello atteso da Terraform:
`rf-distributed-datasets-<ACCOUNT_ID>-us-east-1`.

### 3.2 Log group CloudWatch per ECS

I task definition di orchestrator e worker scrivono i log su CloudWatch
tramite `awslogs`, ma **non impostano `awslogs-create-group`** (creare un
log group al volo dal container va anch'esso in conflitto con la SCP).
Vanno quindi creati a mano, una sola volta:

```bash
aws logs create-log-group --log-group-name "/ecs/lab-orchestrator" --region us-east-1
aws logs create-log-group --log-group-name "/ecs/lab-worker" --region us-east-1
```

Se il gruppo esiste già, il comando restituisce un errore innocuo
(`ResourceAlreadyExistsException`) che puoi ignorare.

### 3.3 Limite di memoria per le task ECS

La SCP nega `ecs:RegisterTaskDefinition` per qualunque task con
**`memory` superiore a 8192 MiB**, indipendentemente da `cpu`, tag o altri
parametri (verificato empiricamente per bisezione). Le variabili
`worker_memory` / `orchestrator_memory` in `variables.tf` sono già impostate
di default a `8192` per questo motivo — **non alzarle** oltre questo valore,
o il deploy fallirà con `AccessDeniedException`.

> Se il tuo pool di processi paralleli lato applicativo (dimensionato
> storicamente per 16 GB) risente della RAM ridotta, valuta di abbassare il
> numero di processi concorrenti nel codice worker invece di alzare la
> memory della task.

### 3.4 Tag obbligatorio sulle risorse ECS

La stessa SCP nega anche la creazione di risorse ECS (task definition,
cluster) se la richiesta non porta **almeno un tag** (il nome/valore non
sembra contare, solo la presenza). Il provider è già configurato con
`default_tags` in `provider.tf`, quindi non serve fare nulla — è documentato
qui solo per chiarezza, nel caso in futuro si tolga quel blocco per errore.

## 4. Deploy

```bash
terraform init
terraform plan    # opzionale, mostra cosa verrà creato
terraform apply
```

Il primo apply richiede qualche minuto in più per il build+push
dell'immagine Docker. Al termine, Terraform stampa i prossimi passi
(output `next_steps`) con i comandi pronti per configurare il `.env`
e avviare un test.

## 5. Eseguire un test

Dopo l'apply, dalla root del progetto (fuori da `terraform/`):

```bash
# aggiorna il tuo .env con i valori mostrati in output (bucket S3, regione, ecc.)
./run_aws.sh                          # avvia il client contro l'infrastruttura
./script_aws/run_test_engine_ecs.sh   # oppure: sessione di test interattiva
```

## 6. Fermare l'esecuzione senza distruggere l'infrastruttura

Creare un `aws_ecs_service` con `desired_count > 0` fa sì che **ECS avvii
subito i task e li mantenga attivi in autonomia** (non serve un comando
separato per "avviare": succede appena il Service viene creato). Per non
consumare crediti quando non stai facendo un run attivo, scala i Service a
zero invece di distruggere tutta l'infrastruttura:

```bash
aws ecs update-service --cluster forest-cluster --service orchestrator-service --desired-count 0 --region us-east-1
aws ecs update-service --cluster forest-cluster --service worker-service --desired-count 0 --region us-east-1
```

Per rendere lo stop persistente anche attraverso un futuro `terraform apply`
(altrimenti Terraform riporterebbe i contatori ai valori dichiarati nel
codice), imposta in `terraform.tfvars`:

```hcl
orchestrator_desired_count = 0
num_workers                = 0
```

Per riavviare un run, rimetti i valori desiderati (es. `orchestrator_desired_count = 2`,
`num_workers = 2`) e rilancia `terraform apply`: le task definition esistono
già, verranno solo referenziate di nuovo dai Service.

**Verifica rapida che non ci sia nulla in esecuzione** (task Fargate è
l'unica risorsa di questo stack che fattura per tempo, non per richiesta):

```bash
aws ecs list-tasks --cluster forest-cluster --region us-east-1
# {"taskArns": []}  → nessun task attivo, nessun consumo di calcolo in corso
```

## 7. Distruggere tutto

```bash
cd terraform
terraform destroy
```

Distrugge le risorse gestite da Terraform (ECS, ECR con l'immagine,
DynamoDB, SQS, Security Group). Da lanciare a fine sessione di valutazione
per non lasciare nulla attivo nel Learner Lab.

> ⚠️ Il **bucket S3** creato manualmente al punto 3.1 (e i **log group**
> CloudWatch del punto 3.2) sono referenziati da Terraform come risorse
> esistenti, non creati da esso — `terraform destroy` **non li elimina**.
> Se vuoi ripulirli del tutto:
> ```bash
> aws s3 rb s3://rf-distributed-datasets-<ACCOUNT_ID>-us-east-1 --force
> aws logs delete-log-group --log-group-name "/ecs/lab-orchestrator" --region us-east-1
> aws logs delete-log-group --log-group-name "/ecs/lab-worker" --region us-east-1
> ```

## Note di design

- **Nessuna risorsa IAM viene creata**: il modulo referenzia il ruolo
  `LabRole` già presente in ogni account Learner Lab (`data.aws_iam_role`).
  Se lanciato fuori da un Learner Lab, `LabRole` non esiste e va sostituito
  con un `aws_iam_role` equivalente.
- **VPC**: viene riusata quella di default dell'account/regione (sempre
  presente), non ne viene creata una nuova.
- **Modalità `federated`**: prima di sottomettere un job, va eseguito
  `provision_federated_shards.py` per popolare gli shard su S3 (vedi
  output `next_steps` dopo l'apply).
- **Rebuild dell'immagine**: avviene automaticamente solo se cambiano
  `Dockerfile` o file sotto `src/` (hash calcolato nei `triggers` di
  `docker_build.tf`). Per forzare sempre il rebuild, imposta
  `force_image_rebuild = true` in `terraform.tfvars`.
- **Restrizioni SCP del Learner Lab**: vedi la sezione
  [Setup manuale richiesto](#3-setup-manuale-richiesto-solo-learner-lab)
  per l'elenco completo e le motivazioni verificate empiricamente.