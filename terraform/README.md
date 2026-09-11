# Infrastruttura AWS via Terraform — Distributed_RandomForest

Questo modulo Terraform crea **da zero** tutta l'infrastruttura AWS necessaria
al sistema (ECR, S3, DynamoDB, SQS, ECS Fargate + Service orchestrator/worker),
buildando e pushando anche l'immagine Docker dell'applicazione. Pensato per
essere eseguito con un **singolo `terraform apply`** in un account
**AWS Academy Learner Lab**.

> Un Learner Lab impone alcune restrizioni particolari (SCP) che richiedono
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

> ⚠️ Questo bucket è creato **fuori** da Terraform: se l'account Lab viene
> resettato o ricreato (Start/End Lab, non un semplice refresh delle
> credenziali), il bucket sparisce insieme a tutto il resto e va ricreato da
> capo con lo stesso comando prima del prossimo `apply` — `terraform plan`
> non lo segnala come mancante (lo referenzia come risorsa già esistente),
> quindi se te lo dimentichi il fallimento si presenta più avanti, al primo
> salvataggio di un dataset, non durante l'apply stesso.

### 3.2 Log group CloudWatch

I task/istanze di orchestrator, worker e test-engine scrivono i log su
CloudWatch tramite `awslogs`, ma **non impostano `awslogs-create-group`**
(creare un log group al volo va anch'esso in conflitto con la SCP).
Vanno quindi creati a mano, una sola volta per account:

```bash
# Worker (ECS Fargate) e log storico dell'orchestrator quando era su ECS
aws logs create-log-group --log-group-name "/ecs/lab-orchestrator" --region us-east-1
aws logs create-log-group --log-group-name "/ecs/lab-worker" --region us-east-1

# Orchestrator EC2 (vedi orchestrator_ec2.tf) e test-engine EC2 on-demand
# (vedi run_test_engine.sh) - entrambi mancanti in una versione precedente
# di questa sezione, scoperti solo al primo setup su un account nuovo.
aws logs create-log-group --log-group-name "/ec2/lab-orchestrator" --region us-east-1
aws logs create-log-group --log-group-name "/ec2/rf-test-engine" --region us-east-1
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

> ⚠️ **L'apply crea l'infrastruttura ma la lascia ferma** (vedi sezione 6):
> nessun worker né istanza orchestrator è in esecuzione subito dopo un
> `apply` pulito. Avvia entrambi prima di lanciare qualunque test — un job
> inviato a un'infrastruttura ferma resta semplicemente in coda SQS senza
> che nessuno lo reclami, senza un errore esplicito che lo segnali.

> ⚠️ **Prima di lanciare qualunque script, aggiorna `API_GATEWAY_URL` nel
> `.env`** con il valore mostrato nell'output `next_steps` dell'apply appena
> fatto. Questo endpoint **cambia a ogni ricreazione dello stack** (nuovo
> apply dopo un `destroy`, o dopo un reset dell'account Lab): se lasci il
> valore vecchio, il client non fallisce in modo esplicito all'avvio — parla
> semplicemente con un endpoint API Gateway che non esiste più (o che
> appartiene a un deploy precedente), quindi il sintomo è una richiesta che
> non arriva mai a destinazione, non un errore chiaro. Vale anche per il
> bucket S3 e la region, se sono cambiati.

```bash
# aggiorna il tuo .env con i valori mostrati in output (bucket S3, regione,
# e soprattutto API_GATEWAY_URL — vedi avviso sopra)
./run_aws.sh                          # avvia il client contro l'infrastruttura
./script_aws/run_test_engine_ecs.sh   # oppure: sessione di test interattiva
```

## 6. Avviare e fermare l'esecuzione senza distruggere l'infrastruttura

Dalla versione corrente, **un `apply` pulito crea tutte le risorse ma le
lascia ferme**: sia `orchestrator_desired_count` sia `worker_desired_count`
hanno default `0` in `variables.tf`. Nessun task Fargate né istanza EC2
dell'orchestrator parte da sola subito dopo l'apply — un passo esplicito è
sempre richiesto, in entrambe le modalità.

> Se vieni da una versione precedente del progetto: `num_workers` ora
> controlla **solo** quante risorse esistono (task definition/service, per
> federated anche gli indici) — non più quante sono avviate. Per quello
> serve `worker_desired_count` (vedi sotto). Vedi anche la sezione
> [8. Passare tra centralized e federated](#8-passare-tra-centralized-e-federated).

### 6.1 Avviare

**Orchestrator** (2 istanze EC2, sempre uguale in entrambe le modalità):
```bash
aws autoscaling update-auto-scaling-group --auto-scaling-group-name orchestrator-asg \
  --min-size 2 --max-size 2 --desired-capacity 2 --region us-east-1
```

**Worker, modalità `centralized`** (un solo service):
```bash
aws ecs update-service --cluster forest-cluster --service worker-service \
  --desired-count 10 --region us-east-1   # 10 = quanti vuoi avviare, fino a num_workers
```

**Worker, modalità `federated`** (N service separati, uno per indice — ognuno
ospita al massimo un solo task, quindi qui non "quanti" ma "tutti o nessuno"):
```bash
for i in $(seq 1 10); do   # 10 = num_workers
  aws ecs update-service --cluster forest-cluster --service "worker-service-$i" \
    --desired-count 1 --region us-east-1 > /dev/null
done
```

Aspetta qualche minuto dopo l'avvio prima di lanciare un test (boot EC2,
pull immagine, avvio container) — vedi le verifiche già usate altrove in
questo progetto (`storedBytes` su CloudWatch Logs, `docker ps` via SSM).

### 6.2 Fermare

Stessi comandi di sopra con `--desired-count 0` / `--min-size 0 --max-size 0
--desired-capacity 0` (loop identico per i worker federated).

**Verifica rapida che non ci sia nulla in esecuzione** (task Fargate e
istanze EC2 sono le uniche risorse di questo stack che fatturano per tempo,
non per richiesta):
```bash
aws ecs list-tasks --cluster forest-cluster --region us-east-1
# {"taskArns": []}  → nessun task Fargate attivo
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names orchestrator-asg \
  --query 'AutoScalingGroups[0].Instances' --region us-east-1
# []  → nessuna istanza orchestrator attiva
```

### 6.3 Rendere lo stop persistente attraverso un futuro `apply`

I comandi sopra agiscono solo sullo stato attuale in AWS — un successivo
`terraform apply` riporterebbe i contatori ai valori dichiarati in
`terraform.tfvars`. Per farlo restare fermo anche dopo un apply:

```hcl
orchestrator_desired_count = 0
worker_desired_count       = 0   # NON num_workers: quello controlla solo
                                  # quante risorse esistono, non quante girano
```
(sono già i default se non li specifichi affatto in `terraform.tfvars`.)

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

## 8. Passare tra `centralized` e `federated`

Le due modalità **non sono intercambiabili a runtime**: cambiano quali
risorse Terraform crea, quindi il passaggio richiede un `terraform apply`
vero e proprio, non solo una modifica al `.env` locale.

### 8.1 Cosa modificare, e cosa NON basta

```hcl
# terraform.tfvars
training_mode = "federated"   # o "centralized"
```

> ⚠️ **Modificare `TRAINING_MODE` nel `.env` locale da solo NON è
> sufficiente.** Il `.env` controlla solo il client e il test-engine (le
> istanze EC2 usa-e-getta, che leggono la variabile a ogni lancio) — ma i
> worker ECS già deployati hanno `TRAINING_MODE` **cablato staticamente**
> nella loro `container_definitions` (vedi `ecs_task_definitions.tf`,
> `local.common_env`), fissato al valore di `var.training_mode` al momento
> dell'ultimo `apply`. Senza rifare l'`apply`, il test-engine proverebbe a
> orchestrare l'altra modalità parlando con worker che si aspettano ancora
> quella vecchia — protocollo/logica di partizionamento incompatibili.

Dopo aver cambiato `training_mode`:

```bash
cd terraform
terraform plan -out=tfplan     # controlla il piano PRIMA di applicare:
                                # cambiare modalità distrugge il service/le
                                # task definition della modalità precedente
terraform apply "tfplan"
```

### 8.2 Cosa cambia concretamente nell'infrastruttura

| | `centralized` | `federated` |
|---|---|---|
| Task definition worker | 1 (`lab-worker-task`) | N, una per indice (`lab-worker-task-1` … `lab-worker-task-N`) |
| Service ECS worker | 1 solo (`worker-service`), `desired_count = worker_desired_count` | N service separati (`worker-service-1` … `worker-service-N`), ciascuno `desired_count = worker_desired_count > 0 ? 1 : 0` (avvio tutto-o-niente, non parziale per indice) |
| Ruolo dei worker | anonimi, intercambiabili | indice fisso 1..N, legato al proprio shard (`WORKER_INDEX` iniettato staticamente da Terraform) |
| `num_workers` significa | quante risorse esistono (non più quante sono avviate — vedi sezione 6) | quanti indici/shard fissi creare |

Per avviare/fermare i worker in entrambe le modalità, vedi la sezione
[6. Avviare e fermare l'esecuzione](#6-avviare-e-fermare-lesecuzione-senza-distruggere-linfrastruttura),
che copre già sia `centralized` sia `federated`. Un comando utile solo per
`federated`, per vedere lo stato di tutti gli indici in un colpo solo:

```bash
aws ecs describe-services --cluster forest-cluster \
  --services $(for i in $(seq 1 10); do echo -n "worker-service-$i "; done) \
  --region us-east-1 --query 'services[].[serviceName,status,runningCount,desiredCount]' --output table
```

### 8.3 Provisioning dati: solo per il dataset reale

Se lavori con `dataset_type=synthetic`, **salta questo passo**: i dati
vengono generati al volo al primo training, nessun file va pre-caricato.

Solo per `dataset_type=real` (partizionamento `by_day` su CICIDS), esegui
**prima** di sottomettere un job:

```bash
python -m scripts.provision_federated_shards --num-workers N
```

con lo stesso `N` di `num_workers` in `terraform.tfvars` — un disallineamento
tra i due lascerebbe worker senza shard assegnato (vedi il fallback a
"dimensione 0, worker saltato nel round" descritto in
`federatedWorker.py::exposed_get_local_shard_size`).

## Note di design

- **Nessuna risorsa IAM viene creata**: il modulo referenzia il ruolo
  `LabRole` già presente in ogni account Learner Lab (`data.aws_iam_role`).
  Se lanciato fuori da un Learner Lab, `LabRole` non esiste e va sostituito
  con un `aws_iam_role` equivalente.
- **VPC**: viene riusata quella di default dell'account/regione (sempre
  presente), non ne viene creata una nuova.
- **Modalità `federated`**: il provisioning degli shard
  (`provision_federated_shards.py`) serve **solo** per `dataset_type=real`
  (partizionamento `by_day` su S3, letto da ogni worker al boot). Per
  `dataset_type=synthetic` **non va eseguito**: i dati vengono generati
  pigramente al primo training, non richiedono nulla pre-caricato su S3
  (vedi `federatedWorker.py::exposed_get_local_shard_size`, che gestisce
  esplicitamente l'assenza dello shard file per il sintetico). Vedi anche
  la sezione [8. Passare tra centralized e federated](#8-passare-tra-centralized-e-federated).
- **Rebuild dell'immagine**: avviene automaticamente solo se cambiano
  `Dockerfile` o file sotto `src/` (hash calcolato nei `triggers` di
  `docker_build.tf`). Per forzare sempre il rebuild, imposta
  `force_image_rebuild = true` in `terraform.tfvars`.
- **Restrizioni SCP del Learner Lab**: vedi la sezione
  [Setup manuale richiesto](#3-setup-manuale-richiesto-solo-learner-lab)
  per l'elenco completo e le motivazioni verificate empiricamente.