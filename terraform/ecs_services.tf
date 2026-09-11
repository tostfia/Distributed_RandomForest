locals {
  network_config_subnets = data.aws_subnets.public.ids
}

# ---------------------------------------------------------------------
# ORCHESTRATOR: rimosso da qui, non gira più come Service ECS Fargate.
# Ora gira su istanze EC2 dedicate (aws_instance.orchestrator) — vedi
# orchestrator_ec2.tf per il dettaglio e la motivazione (tetto di memoria
# SCP del Learner Lab).
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# WORKER SERVICE - modalità CENTRALIZED: un unico Service, desired_count
# = num_workers, worker anonimi e intercambiabili.
# ---------------------------------------------------------------------
resource "aws_ecs_service" "worker_centralized" {
  count = var.training_mode == "centralized" ? 1 : 0

  name            = "worker-service"
  cluster         = aws_ecs_cluster.forest_cluster.id
  task_definition = aws_ecs_task_definition.worker_centralized[0].arn
  # Disaccoppiato da num_workers (che ora controlla solo quante risorse
  # ESISTONO, non quante sono avviate): default 0, l'apply crea il service
  # ma non fa partire nessun task Fargate finché non alzi questa variabile.
  desired_count   = var.worker_desired_count
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = local.network_config_subnets
    security_groups  = [aws_security_group.rf_distributed.id]
    assign_public_ip = true
  }

  force_new_deployment = true

  tags = { Project = var.project_name }
}

# ---------------------------------------------------------------------
# WORKER SERVICE - modalità FEDERATED: un Service per ciascun indice
# fisso, desired_count=1 ciascuno (un solo task per indice/shard: se ne
# morisse uno, ECS lo riavvia con LO STESSO indice, mantenendo il binding
# worker<->shard su S3).
# ---------------------------------------------------------------------
resource "aws_ecs_service" "worker_federated" {
  count = var.training_mode == "federated" ? var.num_workers : 0

  name            = "worker-service-${count.index + 1}"
  cluster         = aws_ecs_cluster.forest_cluster.id
  task_definition = aws_ecs_task_definition.worker_federated[count.index].arn
  # Non più fisso a 1: gated su worker_desired_count (default 0), stessa
  # logica di worker_centralized sopra. Ogni service federated ospita
  # comunque al massimo un solo task (un worker per indice/shard fisso),
  # quindi qualunque valore > 0 qui equivale a "avviato" per TUTTI i 
  # service insieme - non esiste un avvio parziale per singolo indice
  # tramite questa variabile (vedi variables.tf per il dettaglio).
  desired_count   = var.worker_desired_count > 0 ? 1 : 0
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = local.network_config_subnets
    security_groups  = [aws_security_group.rf_distributed.id]
    assign_public_ip = true
  }

  force_new_deployment = true

  tags = { Project = var.project_name }
}