# Infrastruttura AWS via Terraform — Distributed_RandomForest

Questo modulo Terraform crea **da zero** tutta l'infrastruttura AWS necessaria
al sistema (ECR, S3, DynamoDB, SQS, ECS Fargate + Service orchestrator/worker),
buildando e pushando anche l'immagine Docker dell'applicazione. Pensato per
essere eseguito con un **singolo `terraform apply`** in un account
**AWS Academy Learner Lab**.

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

## 2. Configurazione

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# modifica terraform.tfvars se necessario (es. training_mode, num_workers)
```

## 3. Deploy

```bash
terraform init
terraform plan    # opzionale, mostra cosa verrà creato
terraform apply
```

Il primo apply richiede qualche minuto in più per il build+push
dell'immagine Docker. Al termine, Terraform stampa i prossimi passi
(output `next_steps`) con i comandi pronti per configurare il `.env`
e avviare un test.

## 4. Eseguire un test

Dopo l'apply, dalla root del progetto (fuori da `terraform/`):

```bash
# aggiorna il tuo .env con i valori mostrati in output (bucket S3, regione, ecc.)
./run_aws.sh                    # avvia il client contro l'infrastruttura
./script_aws/run_test_engine_ecs.sh   # oppure: sessione di test interattiva
```

## 5. Distruggere tutto

```bash
cd terraform
terraform destroy
```

Distrugge **tutte** le risorse create da Terraform (ECS, ECR con l'immagine,
S3 con il suo contenuto, DynamoDB, SQS, Security Group). Da lanciare a fine
sessione di valutazione per non lasciare nulla attivo nel Learner Lab.

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
