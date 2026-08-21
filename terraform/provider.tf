# ---------------------------------------------------------------------
# Provider AWS.
#
# In AWS Academy Learner Lab le credenziali sono TEMPORANEE (access key +
# secret key + session token, scadono ogni ~4h) e vanno prese dalla scheda
# "AWS Details" del Lab. Terraform le legge, come la AWS CLI, da una di
# queste fonti (in ordine di precedenza standard):
#   1. Variabili d'ambiente AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
#      AWS_SESSION_TOKEN (consigliato: copia/incolla dal Lab prima di
#      lanciare terraform apply/destroy)
#   2. File ~/.aws/credentials, profilo [default]
#
# Non serve configurare nulla qui: il provider le raccoglie da sole.
# Se durante l'apply ricevi errori di autenticazione/scadenza token,
# aggiorna le credenziali dal Learner Lab e rilancia (terraform riprende
# da dove si era fermato grazie allo state).
# ---------------------------------------------------------------------
provider "aws" {
  region = var.aws_region
}
