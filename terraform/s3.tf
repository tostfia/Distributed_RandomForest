# Nome globalmente unico: costruito dinamicamente con l'account id, così
# funziona senza modifiche in QUALSIASI account Learner Lab (il vecchio
# bucket 'my-cluster-datasets-bucket-759804778194-...' aveva l'account id
# della TUA sessione hardcoded nel nome di default in deploy.sh/config.py:
# qui invece si autoadatta).
resource "aws_s3_bucket" "datasets" {
  bucket = "${var.project_name}-datasets-${data.aws_caller_identity.current.account_id}-${var.aws_region}"

  tags = {
    Project = var.project_name
  }
}

# Versioning disattivato: verificato con 'get-bucket-versioning' sul bucket
# esistente, risposta vuota = non configurato/disabilitato. Nessuna
# necessità di versioning per dataset/modelli rigenerabili.
resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  versioning_configuration {
    status = "Disabled"
  }
}

# Blocco accesso pubblico: il bucket contiene dataset e modelli, nessuna
# ragione per essere raggiungibile da internet.
resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
