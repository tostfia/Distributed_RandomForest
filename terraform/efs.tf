# =============================================================================
# EFS come cache di lettura condivisa per il dataset di training (11/9/2026)
# =============================================================================
# PROBLEMA MISURATO: in modalita' centralized, ogni worker scarica l'INTERO
# dataset condiviso da S3 (fino a ~1.5 GB), anche quando N worker fanno tutti
# lo stesso identico download per lo stesso job - costo di rete fisso per
# worker, misurato in ~16s a testa l'11/9/2026 (vedi dataset_dao.py, timing
# strumentato). Con N worker, significa N download ridondanti dello stesso
# file.
#
# SOLUZIONE: l'ORCHESTRATOR (unico scrittore, elimina ogni race condition)
# scrive il dataset ANCHE su un filesystem EFS condiviso, subito dopo l'ETL,
# oltre al normale upload S3 (che resta l'URL canonico richiesto dalla
# traccia d'esame - nessun cambiamento li'). I worker Fargate montano lo
# stesso EFS e LEGGONO da li' se la copia e' presente, con fallback su S3
# se per qualunque motivo non lo e' (mount fallito, EFS non ancora
# popolato, ecc.) - mai un singolo point of failure.
#
# NON usato in modalita' federated: li' i worker leggono ciascuno il proprio
# shard (nessuna ridondanza tra worker sullo stesso file), il problema che
# EFS risolve qui non si applica.

resource "aws_efs_file_system" "dataset_cache" {
  creation_token = "${var.project_name}-dataset-cache"

  # Bursting (default): throughput proporzionale allo storage occupato.
  # Per un dataset di poche centinaia di MB - GB il tetto puo' essere
  # basso rispetto a S3 (che scala bene su letture sequenziali grandi) -
  # vedi discussione: valutare 'provisioned' con un throughput esplicito
  # se il beneficio misurato risultasse inferiore alle attese, prima di
  # dare per scontato che EFS batta sempre S3 su throughput puro.
  throughput_mode = "bursting"

  encrypted = true

  tags = {
    Project = var.project_name
  }
}

# Un mount target per ogni subnet pubblica dove un worker Fargate puo'
# atterrare (stesso insieme di subnet gia' usato dai worker in
# ecs_services.tf, network_config_subnets = data.aws_subnets.public.ids):
# senza un mount target nella AZ del worker, il mount fallirebbe per
# quel worker specifico.
resource "aws_efs_mount_target" "dataset_cache" {
  for_each = toset(data.aws_subnets.public.ids)

  file_system_id  = aws_efs_file_system.dataset_cache.id
  subnet_id       = each.value
  security_groups = [aws_security_group.rf_distributed.id]
}

# NFS (porta 2049) tra il security group condiviso e se stesso: orchestrator
# e worker sono entrambi in questo stesso SG (vedi ecs_services.tf,
# orchestrator_ec2.tf), quindi una singola regola self-referencing copre
# sia la scrittura (orchestrator) sia la lettura (worker).
resource "aws_security_group_rule" "efs_nfs_self_ingress" {
  type              = "ingress"
  from_port         = 2049
  to_port           = 2049
  protocol          = "tcp"
  security_group_id = aws_security_group.rf_distributed.id
  self              = true
  description       = "NFS per il mount EFS della cache dataset (orchestrator scrive, worker leggono)"
}