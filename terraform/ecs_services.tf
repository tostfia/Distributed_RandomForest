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
  desired_count   = var.num_workers
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
  desired_count   = 1
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