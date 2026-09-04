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
    { name = "DATASETS_BUCKET_NAME", value = data.aws_s3_bucket.datasets.bucket },
  ]
}

# ---------------------------------------------------------------------
# ORCHESTRATOR (unica family, sempre presente in entrambe le modalità).
# ORCHESTRATOR_INDEX viene derivato a runtime dal Task ARN via ECS
# Container Metadata Endpoint, identico a deploy.sh: è ciò che permette
# a due istanze orchestrator di distinguersi per la leader election
# senza dover essere ri-configurate a mano.
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "orchestrator" {
  family                   = "rf-orchestrator-task"
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
          "awslogs-group"         = "/ecs/rf-orchestrator"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "orchestrator"
          "awslogs-create-group"  = "true"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}

# ---------------------------------------------------------------------
# WORKER - modalità CENTRALIZED: un'unica family, worker anonimi e
# intercambiabili, scalati tramite desired_count sul Service.
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "worker_centralized" {
  count = var.training_mode == "centralized" ? 1 : 0

  family                   = "rf-worker-task"
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
          "awslogs-group"         = "/ecs/rf-worker"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "worker"
          "awslogs-create-group"  = "true"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}

# ---------------------------------------------------------------------
# WORKER - modalità FEDERATED: una family per ciascun indice fisso
# 1..num_workers (binding worker<->shard, coerente con quanto richiesto
# da provision_federated_shards.py). WORKER_INDEX è iniettato staticamente
# nella Task Definition, non derivato a runtime come per l'orchestrator,
# perché ogni indice deve mappare SEMPRE sullo stesso shard S3.
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "worker_federated" {
  count = var.training_mode == "federated" ? var.num_workers : 0

  family                   = "rf-worker-task-${count.index + 1}"
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
          "awslogs-group"         = "/ecs/rf-worker"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "worker-${count.index + 1}"
          "awslogs-create-group"  = "true"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}

# ---------------------------------------------------------------------
# TEST ENGINE - task one-off, lanciato con `aws ecs run-task` da
# run_test_engine_ecs.sh. Qui registriamo solo la Task Definition
# "template": lo scenario da eseguire (variabile SCENARIO) NON è fissato
# qui, ma passato a runtime dallo script tramite --overrides in
# run-task, così un singolo terraform apply basta per tutti gli scenari.
# ---------------------------------------------------------------------
resource "aws_ecs_task_definition" "test_engine" {
  family                   = "rf-test-engine-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.test_engine_cpu
  memory                   = var.test_engine_memory
  task_role_arn             = data.aws_iam_role.lab_role.arn
  execution_role_arn        = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "test-engine"
      image     = local.image_uri
      essential = true
      environment = concat(local.common_env, [
        { name = "RPC_SYNC_TIMEOUT_SECONDS", value = "${var.rpc_sync_timeout_seconds}s" },
        { name = "RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", value = "${var.rpc_inference_sync_timeout_seconds}s" },
        # Valore di default; run_test_engine_ecs.sh lo sovrascrive a
        # runtime con --overrides in base allo scenario scelto.
        { name = "SCENARIO", value = "all" },
      ])
      command = ["sh", "-c", "timeout 7200 python -m src.testing.engine"]
      linuxParameters = {
        initProcessEnabled = true
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/rf-test-engine"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "test-engine"
          "awslogs-create-group"  = "true"
        }
      }
    }
  ])

  depends_on = [null_resource.docker_build_push]
}
