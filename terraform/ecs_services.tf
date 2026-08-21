locals {
  network_config_subnets = data.aws_subnets.public.ids
}

# ---------------------------------------------------------------------
# ORCHESTRATOR SERVICE (sempre presente).
#
# deployment_minimum_healthy_percent=0 / maximum_percent=100: forza ECS a
# spegnere i task vecchi PRIMA di avviare quelli nuovi durante un
# deployment, niente overlap temporaneo vecchi+nuovi che confonderebbe la
# leader election (stessa scelta motivata in deploy.sh).
# ---------------------------------------------------------------------
resource "aws_ecs_service" "orchestrator" {
  name            = "orchestrator-service"
  cluster         = aws_ecs_cluster.forest_cluster.id
  task_definition = aws_ecs_task_definition.orchestrator.arn
  desired_count   = var.orchestrator_desired_count
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = local.network_config_subnets
    security_groups  = [aws_security_group.rf_distributed.id]
    assign_public_ip = true
  }

  # Il numero di worker/il codice dell'app possono cambiare (nuova build
  # immagine): forziamo un nuovo deployment ogni volta che la Task
  # Definition cambia revisione, invece di lasciare i vecchi task in vita.
  force_new_deployment = true

  tags = { Project = var.project_name }
}

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
