# ---------------------------------------------------------------------
# Identità account: usata per rendere univoco il nome del bucket S3 e per
# costruire l'URL del registry ECR, in modo che il modulo funzioni
# identico in QUALSIASI account Learner Lab (docenti inclusi) senza
# hardcodare nessun account id.
# ---------------------------------------------------------------------
data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# ---------------------------------------------------------------------
# LabRole: in AWS Academy Learner Lab non è possibile creare ruoli o
# policy IAM (permesso negato su iam:CreateRole/CreatePolicy). Esiste
# però già un ruolo precreato "LabRole" con i permessi necessari (ECS,
# S3, DynamoDB, SQS, ECR, CloudWatch Logs): lo referenziamo soltanto,
# esattamente come fa deploy.sh con LABROLE_ARN.
#
# Se lanci questo modulo FUORI da un Learner Lab (account AWS normale),
# LabRole non esiste: dovrai creare un ruolo equivalente a mano e
# sostituire questo data source con una risorsa aws_iam_role.
# ---------------------------------------------------------------------
data "aws_iam_role" "lab_role" {
  name = "LabRole"
}

# ---------------------------------------------------------------------
# VPC di default: i Learner Lab non permettono di creare VPC personalizzate
# in molti casi, e comunque non serve: la VPC di default è già presente in
# ogni account/regione e deploy.sh la usa già così.
# ---------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

# Solo le subnet pubbliche (assegnano IP pubblico in automatico): sono
# quelle usate per i task Fargate con assign_public_ip = true, dato che
# il Learner Lab non fornisce NAT Gateway di default per le subnet private.
data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}
