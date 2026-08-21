resource "aws_security_group" "rf_distributed" {
  name        = "${var.project_name}-sg"
  description = "Orchestrator-Worker RPC + SSH/debug"
  vpc_id      = data.aws_vpc.default.id

  tags = {
    Project = var.project_name
  }
}

# Regola self-referencing: qualunque risorsa associata a questo stesso SG
# (worker, orchestrator, test-engine) può raggiungere le altre sulla porta
# RPC, senza esporre nulla verso l'esterno. Identica a quella creata da
# deploy.sh con 'aws ec2 authorize-security-group-ingress --source-group'.
resource "aws_security_group_rule" "rpc_self_ingress" {
  type                     = "ingress"
  from_port                = var.rpc_port
  to_port                  = var.rpc_port
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rf_distributed.id
  source_security_group_id = aws_security_group.rf_distributed.id
  description               = "RPC self-referencing tra orchestrator/worker/test-engine"
}

# Egress libero: necessario per il pull dell'immagine da ECR, accesso a
# S3/DynamoDB/SQS via endpoint pubblici, e chiamate al SSM Agent per
# ECS Exec (usato da run_test_engine_ecs.sh).
resource "aws_security_group_rule" "egress_all" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.rf_distributed.id
  description       = "Egress libero (pull immagini ECR, S3/DynamoDB/SQS, SSM per ECS Exec)"
}
