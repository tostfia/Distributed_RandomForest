locals {
  image_uri = "${aws_ecr_repository.rf_distributed.repository_url}:${var.image_tag}"

  # Variabili d'ambiente comuni a orchestrator e worker.
  common_env = [
    { name = "PYTHONDONTWRITEBYTECODE", value = "1" },
    { name = "PYTHONUNBUFFERED", value = "1" },
    { name = "NUM_WORKERS", value = tostring(var.num_workers) },
    { name = "ENV_MODE", value = "aws" },
    { name = "SYS_ENV", value = "aws" },
    { name = "TRAINING_MODE", value = var.training_mode },
    { name = "SYS_MODE", value = var.training_mode },
    { name = "EC2_ID", value = "Fargate" },
    { name = "RUNNING_IN_DOCKER", value = "true" },
    { name = "AWS_DEFAULT_REGION", value = var.aws_region },
    # Riferimento alla variabile locale definita in s3.tf
    { name = "DATASETS_BUCKET_NAME", value = local.datasets_bucket_name },
  ]
}
# ---------------------------------------------------------------------
# ORCHESTRATOR 
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "orchestrator" {
  family                   = "lab-orchestrator-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.orchestrator_cpu
  memory                   = var.orchestrator_memory
  task_role_arn             = data.aws_iam_role.lab_role.arn
  execution_role_arn        = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "orchestrator"
      image     = local.image_uri
      essential = true
      environment = concat(local.common_env, [
        { name = "WORKER_HEARTBEAT_TIMEOUT", value = var.worker_heartbeat_timeout },
        { name = "RPC_SYNC_TIMEOUT_SECONDS", value = "${var.rpc_sync_timeout_seconds}s" },
        { name = "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", value = "${var.rpc_inference_sync_timeout_seconds}s" },
      ])
      command = [
        "sh", "-c",
        "export ORCHESTRATOR_INDEX=$(curl -s \"$ECS_CONTAINER_METADATA_URI_V4/task\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['TaskARN'].split('/')[-1])\"); echo \"Registrazione con Task ID: $ORCHESTRATOR_INDEX\"; exec python -m src.master.orchestrator.main"
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/lab-orchestrator"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "orchestrator"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}

# ---------------------------------------------------------------------
# WORKER - modalità CENTRALIZED
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "worker_centralized" {
  count = var.training_mode == "centralized" ? 1 : 0

  # Family distinta da quella dell'orchestrator: evita la race condition
  # "Too many concurrent attempts to create a new revision of the specified
  # family" quando Terraform registra entrambe le task definition in
  # parallelo (nessuna dipendenza tra le due risorse). Verificato che la
  # SCP del Learner Lab NON restringe la family a un valore fisso (test
  # empirico del 5/9/2026: registrazione di una family arbitraria riuscita
  # con 'aws ecs register-task-definition --family lab-worker-task-test').
  family                   = "lab-worker-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  task_role_arn             = data.aws_iam_role.lab_role.arn
  execution_role_arn        = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = local.image_uri
      essential = true
      environment = concat(local.common_env, [
        { name = "WORKER_HEARTBEAT_TIMEOUT", value = var.worker_heartbeat_timeout },
        { name = "RPC_PORT", value = tostring(var.rpc_port) },
      ])
      command = [
        "sh", "-c",
        "export RPC_ADVERTISE_HOST=$(curl -s \"$ECS_CONTAINER_METADATA_URI_V4\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['Networks'][0]['IPv4Addresses'][0])\"); echo \"Registrazione con IP: $RPC_ADVERTISE_HOST\"; exec python -m src.worker.main Worker-$${EC2_ID}-$${TRAINING_MODE}-$(hostname) ${var.rpc_port} ${var.training_mode} aws"
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/lab-worker"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "worker"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}

# ---------------------------------------------------------------------
# WORKER - modalità FEDERATED
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "worker_federated" {
  count = var.training_mode == "federated" ? var.num_workers : 0

  # Stessa family per ogni indice, distinta da quella dell'orchestrator
  # (coerente col worker centralized sopra): vedi la nota lì per il motivo
  # del cambio rispetto alla versione precedente.
  family                   = "lab-worker-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  task_role_arn             = data.aws_iam_role.lab_role.arn
  execution_role_arn        = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = local.image_uri
      essential = true
      environment = concat(local.common_env, [
        { name = "WORKER_HEARTBEAT_TIMEOUT", value = var.worker_heartbeat_timeout },
        { name = "WORKER_INDEX", value = tostring(count.index + 1) },
        { name = "RPC_PORT", value = tostring(var.rpc_port) },
      ])
      command = [
        "sh", "-c",
        "export RPC_ADVERTISE_HOST=$(curl -s \"$ECS_CONTAINER_METADATA_URI_V4\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['Networks'][0]['IPv4Addresses'][0])\"); echo \"Registrazione con IP: $RPC_ADVERTISE_HOST (WORKER_INDEX=${count.index + 1})\"; exec python -m src.worker.main Worker-$${EC2_ID}-$${TRAINING_MODE}-$(hostname) ${var.rpc_port} ${var.training_mode} aws"
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/lab-worker"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "worker-${count.index + 1}"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}