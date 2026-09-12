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

    # Cache EFS del dataset condiviso (vedi efs.tf): l'orchestrator e'
    # l'UNICO scrittore (i worker Fargate montano lo stesso EFS in sola
    # lettura, vedi ecs_task_definitions.tf). Il mount NON usa 'set -e'
    # per questo blocco specifico ('|| true' esplicito): se fallisse per
    # qualunque motivo (mount target non ancora propagato, problema di
    # rete transitorio), l'orchestrator deve comunque avviarsi - la
    # variabile EFS_MOUNT_PATH_ARG resta vuota, quindi 'docker run' sotto
    # NON passa EFS_MOUNT_PATH al container, e il codice Python ricade
    # automaticamente sul solo S3 (vedi dataset_dao.py). Un mount fallito
    # degrada le prestazioni, non deve mai far fallire il boot.
    yum install -y amazon-efs-utils || true
    mkdir -p /mnt/efs
    EFS_MOUNT_PATH_ARG=""
    if mount -t efs -o tls ${aws_efs_file_system.dataset_cache.id}:/ /mnt/efs; then
      echo "[EFS] Mount riuscito su /mnt/efs."
      EFS_MOUNT_PATH_ARG="-e EFS_MOUNT_PATH=/mnt/efs -v /mnt/efs:/mnt/efs"
    else
      echo "[EFS] [WARN] Mount fallito - l'orchestrator procede SENZA cache EFS (solo S3)."
    fi

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
      -e CENTRALIZED_DATASET_MODE=${var.centralized_dataset_mode} \
      $EFS_MOUNT_PATH_ARG \
      ${local.image_uri} \
      python -m src.orchestrator.main
  EOF
}

# NOTA sul log group: a differenza di ECS (che può fallire su
# 'awslogs-create-group' per la stessa SCP già vista in ecs_task_definitions.tf),
# qui il gruppo va creato a mano UNA VOLTA, come già fai per /ecs/lab-orchestrator
# e /ecs/lab-worker (vedi README.md, punto 3.2):
#   aws logs create-log-group --log-group-name "/ec2/lab-orchestrator" --region us-east-1

# =============================================================================
# FAULT TOLERANCE: da 'count' di aws_instance a Launch Template + Auto Scaling
# Group a capacità fissa (min=max=desired=2).
#
# PERCHÉ: con due sole istanze in coppia attivo/standby, un singolo guasto è
# già coperto dal lock DynamoDB applicativo (vedi BaseOrchestrator._try_
# acquire_leadership), ma finché l'istanza caduta non torna disponibile il
# sistema resta a UN guasto di distanza dal fermo totale. L'ASG minimizza
# questa finestra di esposizione sostituendo automaticamente l'istanza persa,
# per qualunque motivo (crash OS, terminazione manuale, ecc. — non solo
# guasti hardware).
#
# PERCHÉ È SICURO FARLO SENZA TOCCARE IL CODICE APPLICATIVO (verificato):
#   1. Lo user-data (sopra) deriva hostname/IP dai metadata EC2 ad OGNI boot,
#      non ha nulla di hardcoded sull'istanza precedente: una nuova istanza
#      con un nuovo IP si auto-registra correttamente da sola.
#   2. BaseOrchestrator._get_lock_key() usa una chiave FISSA
#      ("global_orchestrator_leader_lock", tabella OrchestratorLocks), non
#      legata a IP/hostname/instance ID — il lock è agnostico rispetto a
#      quale istanza fisica lo detiene.
#   3. Su terminazione "pulita" (quella che l'ASG usa in scale-in/replace),
#      main.py intercetta SIGTERM, deregistra l'orchestratore e rilascia la
#      leadership nel finally di BaseOrchestrator.start() — nessuna entry
#      fantasma in orchestrators_registry nel caso comune.
#
# COSA COPRE OGNI LIVELLO (i tre livelli sono complementari, non ridondanti):
#   - Crash del processo/container -> 'docker run --restart unless-stopped'
#     (già presente nello user-data sopra), la EC2 resta viva.
#   - Guasto hardware/hypervisor (system status check) -> EC2 Auto Recovery,
#     ABILITATO DI DEFAULT su istanze che lo supportano dal 2023 (verificato
#     empiricamente: 'MaintenanceOptions.AutoRecovery' risulta "default" su
#     un'istanza appena lanciata con questo stesso Launch Template). Nessuna
#     configurazione esplicita necessaria: stessa istanza, stesso ID/IP/EBS.
#   - Istanza stoppata/terminata per qualunque altro motivo -> l'ASG la
#     sostituisce con una nuova (IP diverso, gestito correttamente per i
#     punti 1-3 sopra).
#
# PERMESSI VERIFICATI EMPIRICAMENTE SU LabRole (Learner Lab, nessun accesso
# IAM separato): ec2:CreateLaunchTemplate con IamInstanceProfile,
# autoscaling:CreateAutoScalingGroup/DeleteAutoScalingGroup. Il
# service-linked role 'AWSServiceRoleForAutoScaling' risulta già presente
# nell'account (visto in describe-auto-scaling-groups durante il test).
# =============================================================================

resource "aws_launch_template" "orchestrator" {
  name_prefix   = "orchestrator-lt-"
  image_id      = var.orchestrator_ec2_ami
  instance_type = var.orchestrator_ec2_instance_type

  iam_instance_profile {
    name = "LabInstanceProfile"
  }

  # I security group vanno SOLO qui dentro (non anche in
  # vpc_security_group_ids a livello root): con network_interfaces esplicito
  # (richiesto per associate_public_ip_address), AWS rifiuta la richiesta se
  # entrambi sono presenti - errore visto in pratica su CreateAutoScalingGroup:
  # "When a network interface is provided, the security groups must be a
  # part of it".
  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.rf_distributed.id]
  }

  # Un cambio allo user-data crea una nuova versione del Launch Template;
  # è l'ASG (instance_refresh, sotto) a decidere se e come propagarlo alle
  # istanze esistenti — sostituisce il vecchio 'user_data_replace_on_change'
  # che operava direttamente su aws_instance.
  user_data = base64encode(local.orchestrator_user_data)

  tag_specifications {
    resource_type = "instance"
    tags = {
      Project = var.project_name
      Role    = "orchestrator-ec2"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "orchestrator" {
  name             = "orchestrator-asg"
  min_size         = var.orchestrator_desired_count
  max_size         = var.orchestrator_desired_count
  desired_capacity = var.orchestrator_desired_count

  vpc_zone_identifier = data.aws_subnets.orchestrator_capable.ids

  launch_template {
    id      = aws_launch_template.orchestrator.id
    version = "$Latest"
  }

  # Nessun ELB/Target Group qui (l'orchestrator riceve lavoro da SQS, non da
  # un load balancer): l'health check di tipo EC2 marca l'istanza da
  # sostituire solo quando il suo STATO (non i suoi status check di
  # sistema/istanza) diventa stopped/stopping/terminated/shutting-down. Per
  # i guasti di sistema più fini ci pensa comunque l'EC2 Auto Recovery di
  # default descritto sopra, che mantiene la stessa istanza senza bisogno
  # dell'intervento dell'ASG.
  health_check_type = "EC2"

  # Se cambia il Launch Template (es. nuova immagine Docker via un nuovo
  # apply), l'ASG sostituisce le istanze una alla volta, mantenendo sempre
  # almeno un'istanza (l'altra) disponibile durante il rollout — equivalente
  # allo scopo del vecchio 'user_data_replace_on_change', ma senza il
  # downtime totale che un semplice 'terraform apply' su aws_instance
  # avrebbe causato per un breve periodo.
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }

  tag {
    key                 = "Project"
    value               = var.project_name
    propagate_at_launch = true
  }

  tag {
    key                 = "Role"
    value               = "orchestrator-ec2"
    propagate_at_launch = true
  }

  depends_on = [null_resource.docker_build_push, aws_efs_mount_target.dataset_cache]
}

output "orchestrator_asg_name" {
  description = "Nome dell'Auto Scaling Group dell'orchestrator (per describe-auto-scaling-groups o console)."
  value       = aws_autoscaling_group.orchestrator.name
}

output "orchestrator_ec2_instance_ids_command" {
  description = "Le istanze sono ora gestite dall'ASG e il loro ID cambia ad ogni sostituzione: usare questo comando per ottenere gli ID correnti invece di un output statico."
  value       = "aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names ${aws_autoscaling_group.orchestrator.name} --query 'AutoScalingGroups[0].Instances[].InstanceId' --output text --region ${var.aws_region}"
}