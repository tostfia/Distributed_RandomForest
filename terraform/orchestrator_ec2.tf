# =============================================================================
# ORCHESTRATOR su istanze EC2 (invece di ECS Fargate).
#
# PERCHÉ: la SCP del Learner Lab nega 'ecs:RegisterTaskDefinition' per
# qualunque memoria > 8192 MiB, sia su launch type FARGATE sia EC2-backed
# (verificato empiricamente, entrambi respinti con lo stesso explicit deny).
# 'ec2:RunInstances' su un tipo whitelisted (r5.large, 16 GiB) NON è
# soggetto a questa restrizione (verificato con --dry-run). L'orchestratore
# è l'unico componente che soffre il limite di memoria (i worker restano
# invariati su Fargate, 2 GiB ciascuno, mai sotto stress); qui gli diamo
# 16 GiB invece di 8, per un margine reale (~9 GiB liberi con ~7 GiB di
# alberi in RAM in scenari di scalabilità pesanti).
#
# Sostituisce aws_ecs_service.orchestrator / aws_ecs_task_definition.orchestrator
# in ecs_services.tf / ecs_task_definitions.tf, che vanno rimossi o lasciati
# a desired_count=0 per non consumare risorse duplicate.
# =============================================================================

variable "orchestrator_ec2_ami" {
  description = "AMI Amazon Linux 2023 per le istanze EC2 dell'orchestrator. Da verificare/aggiornare per la regione: 'aws ec2 describe-images --owners amazon --filters \"Name=name,Values=al2023-ami-*-x86_64\" \"Name=state,Values=available\" --query \"sort_by(Images,&CreationDate)[-1].ImageId\" --region us-east-1'"
  type        = string
  # Verificata valida per us-east-1 il 5/9/2026 con il comando sopra.
  # Le AMI Amazon Linux vengono aggiornate periodicamente: rilanciare il
  # comando se questo valore inizia a dare errori 'InvalidAMIID.NotFound'.
  default     = "ami-0ac62d2d72afdce51"
}

variable "orchestrator_ec2_instance_type" {
  description = "Tipo di istanza EC2 per l'orchestrator. r5.large (16 GiB) è confermato fuori dalla whitelist restrittiva del Learner Lab; r5.xlarge e m5.xlarge sono invece bloccati (verificato empiricamente)."
  type        = string
  default     = "r5.large"
}

# r5.large non è offerto in TUTTE le AZ di questo account (verificato
# empiricamente: 'RunInstances' fallisce con 'Unsupported' in us-east-1e,
# messaggio d'errore che elenca esplicitamente le AZ supportate). Filtriamo
# le subnet pubbliche escludendo quella AZ, invece di scegliere "alla
# cieca" per indice come faceva la prima versione — evita di dipendere
# dall'ordine (non garantito stabile da AWS) in cui data.aws_subnets.public
# restituisce gli ID.
data "aws_subnets" "orchestrator_capable" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1f"]
  }
}

locals {
  # User-data comune: installa Docker, autentica su ECR, avvia il container
  # orchestrator con le stesse variabili d'ambiente della vecchia task
  # definition ECS (vedi local.common_env in ecs_task_definitions.tf).
  # Il log driver 'awslogs' nativo di Docker (non serve ECS per usarlo)
  # mantiene il flusso 'aws logs tail' invariato per chi già lo usa.
  # CRITICO: senza --hostname esplicito, Docker assegna al container un ID
  # casuale (es. 'a3f8e91b2c44'), scollegato dall'IP reale dell'istanza.
  # orchestrator_fault.py identifica il leader cercando il pattern
  # 'ip-x-x-x-x.ec2.internal' nel nome registrato su DynamoDB — pattern che
  # finisce lì tramite socket.gethostname() chiamato DENTRO il processo
  # Python (vedi main.py: hostname = socket.gethostname(); orchestrator_name
  # = f"...{hostname}..."). Su Fargate questo funziona perché ECS imposta
  # da sé l'hostname del container in quel formato; qui dobbiamo farlo
  # esplicitamente, ricavando l'IP privato REALE dell'istanza (non quello
  # interno del bridge Docker) dai metadata EC2 PRIMA di avviare il
  # container, così _resolve_ec2_instance_by_ip (che confronta con
  # PrivateIpAddress via 'aws ec2 describe-instances') trova una corrispondenza.
  orchestrator_user_data = <<-EOF
    #!/bin/bash
    set -e
    yum install -y docker
    systemctl enable docker
    systemctl start docker

    aws ecr get-login-password --region ${var.aws_region} | \
      docker login --username AWS --password-stdin ${aws_ecr_repository.rf_distributed.repository_url}

    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    PRIVATE_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/local-ipv4)
    CONTAINER_HOSTNAME="ip-$(echo $PRIVATE_IP | tr '.' '-').ec2.internal"

    docker run -d \
      --name orchestrator \
      --restart unless-stopped \
      --hostname "$CONTAINER_HOSTNAME" \
      --log-driver awslogs \
      --log-opt awslogs-region=${var.aws_region} \
      --log-opt awslogs-group=/ec2/lab-orchestrator \
      --log-opt awslogs-create-group=false \
      --log-opt awslogs-stream=orchestrator-$INSTANCE_ID \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e PYTHONUNBUFFERED=1 \
      -e NUM_WORKERS=${var.num_workers} \
      -e ENV_MODE=aws \
      -e TRAINING_MODE=${var.training_mode} \
      -e EC2_ID=EC2Orchestrator \
      -e ORCHESTRATOR_INDEX=$INSTANCE_ID \
      -e RUNNING_IN_DOCKER=true \
      -e AWS_DEFAULT_REGION=${var.aws_region} \
      -e DATASETS_BUCKET_NAME=${local.datasets_bucket_name} \
      -e WORKER_HEARTBEAT_TIMEOUT=${var.worker_heartbeat_timeout} \
      -e RPC_SYNC_TIMEOUT_SECONDS=${var.rpc_sync_timeout_seconds}s \
      -e RPC_INFERENCE_SYNC_TIMEOUT_SECONDS=${var.rpc_inference_sync_timeout_seconds}s \
      ${local.image_uri} \
      python -m src.master.orchestrator.main
  EOF
}

# NOTA sul log group: a differenza di ECS (che può fallire su
# 'awslogs-create-group' per la stessa SCP già vista in ecs_task_definitions.tf),
# qui il gruppo va creato a mano UNA VOLTA, come già fai per /ecs/lab-orchestrator
# e /ecs/lab-worker (vedi README.md, punto 3.2):
#   aws logs create-log-group --log-group-name "/ec2/lab-orchestrator" --region us-east-1

resource "aws_instance" "orchestrator" {
  count = var.orchestrator_desired_count

  ami                    = var.orchestrator_ec2_ami
  instance_type          = var.orchestrator_ec2_instance_type
  subnet_id              = data.aws_subnets.orchestrator_capable.ids[count.index % length(data.aws_subnets.orchestrator_capable.ids)]
  vpc_security_group_ids = [aws_security_group.rf_distributed.id]
  iam_instance_profile   = "LabInstanceProfile"
  associate_public_ip_address = true

  user_data = local.orchestrator_user_data
  # Senza questo, Terraform aggiorna solo l'attributo 'user_data' nello
  # state ma NON ricrea l'istanza già esistente (lo user_data gira solo al
  # primo boot) — esattamente il motivo per cui il bug dell'hostname
  # (fix di IMDSv2 qui sopra) non si sarebbe mai applicato alle istanze già
  # create con un semplice 'terraform apply', servendo invece a terminarle
  # a mano e farle ricreare da zero. Con true, un cambio allo user_data
  # forza la sostituzione automaticamente.
  user_data_replace_on_change = true

  tags = {
    Project = var.project_name
    Role    = "orchestrator-ec2-${count.index + 1}"
  }

  depends_on = [null_resource.docker_build_push]
}

output "orchestrator_ec2_instance_ids" {
  description = "ID delle istanze EC2 dell'orchestrator (per stop/start manuale, vedi README aggiornato)."
  value       = aws_instance.orchestrator[*].id
}

output "orchestrator_ec2_public_ips" {
  description = "IP pubblici delle istanze EC2 dell'orchestrator, utili per controllo diretto/debug (es. SSM Session Manager, non serve SSH: LabRole include ssm.amazonaws.com)."
  value       = aws_instance.orchestrator[*].public_ip
}