# ---------------------------------------------------------------------
# Schema replicato 1:1 da quello verificato via 'aws dynamodb describe-table'
# sull'infrastruttura esistente. Tutte PAY_PER_REQUEST (on-demand): nessuna
# capacità da dimensionare a mano, coerente con un carico intermittente
# tipico di sessioni di test/training.
# ---------------------------------------------------------------------

resource "aws_dynamodb_table" "workers_registry" {
  name         = "workers_registry"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "worker_name"

  attribute {
    name = "worker_name"
    type = "S"
  }

  tags = { Project = var.project_name }
}

resource "aws_dynamodb_table" "orchestrators_registry" {
  name         = "orchestrators_registry"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "orchestrator_name"

  attribute {
    name = "orchestrator_name"
    type = "S"
  }

  tags = { Project = var.project_name }
}

resource "aws_dynamodb_table" "job_locks" {
  name         = "JobLocks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "lock_key"

  attribute {
    name = "lock_key"
    type = "S"
  }

  tags = { Project = var.project_name }
}

resource "aws_dynamodb_table" "model_status" {
  name         = "ModelStatus"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  tags = { Project = var.project_name }
}

resource "aws_dynamodb_table" "orchestrator_locks" {
  name         = "OrchestratorLocks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "lock_key"

  attribute {
    name = "lock_key"
    type = "S"
  }

  tags = { Project = var.project_name }
}

resource "aws_dynamodb_table" "worker_index_locks" {
  name         = "WorkerIndexLocks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "lock_key"

  attribute {
    name = "lock_key"
    type = "S"
  }

  tags = { Project = var.project_name }
}
resource "aws_dynamodb_table" "job_meta_data"{
  name         = "JobMetadata"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  tags = { Project = var.project_name }
}
# ---------------------------------------------------------------------
# WorkerTasks: unica tabella con due Global Secondary Index, confermati
# via describe-table (worker_name-index, job_id-index). La projection
# ALL è confermata per worker_name-index; per job_id-index è ASSUNTA
# identica (non verificata esplicitamente) — se il codice fa query su
# job_id-index leggendo solo attributi proiettati parzialmente, verifica
# questo punto prima della consegna.
# ---------------------------------------------------------------------
resource "aws_dynamodb_table" "worker_tasks" {
  name         = "WorkerTasks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "task_id"

  attribute {
    name = "task_id"
    type = "S"
  }

  attribute {
    name = "job_id"
    type = "S"
  }

  attribute {
    name = "worker_name"
    type = "S"
  }

  global_secondary_index {
    name            = "worker_name-index"
    hash_key        = "worker_name"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "job_id-index"
    hash_key        = "job_id"
    projection_type = "ALL" # ASSUNTO, vedi nota sopra
  }

  tags = { Project = var.project_name }
}
