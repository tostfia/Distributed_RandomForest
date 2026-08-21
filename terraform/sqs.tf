# Impostazioni replicate 1:1 da 'aws sqs get-queue-attributes' sulle code
# esistenti: FIFO, deduplicazione basata sul contenuto, throughput limit
# per-coda (non per-gruppo-messaggio), visibility timeout 600s, retention
# messaggi 24h (86400s).

resource "aws_sqs_queue" "centralized_queue" {
  name                        = "centralized_queue.fifo"
  fifo_queue                  = true
  content_based_deduplication = true
  fifo_throughput_limit       = "perQueue"
  visibility_timeout_seconds  = 600
  message_retention_seconds   = 86400

  tags = { Project = var.project_name }
}

resource "aws_sqs_queue" "federated_queue" {
  name                        = "federated_queue.fifo"
  fifo_queue                  = true
  content_based_deduplication = true
  fifo_throughput_limit       = "perQueue"
  visibility_timeout_seconds  = 600
  message_retention_seconds   = 86400

  tags = { Project = var.project_name }
}
