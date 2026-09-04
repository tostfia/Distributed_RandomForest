output "cluster_name" {
  description = "Nome del cluster ECS creato."
  value       = aws_ecs_cluster.forest_cluster.name
}

output "ecr_repository_url" {
  description = "URL del repository ECR contenente l'immagine Docker."
  value       = aws_ecr_repository.rf_distributed.repository_url
}

output "datasets_bucket_name" {
  description = "Nome del bucket S3 creato per dataset/shard/modelli/report. Da usare come DATASETS_BUCKET_NAME nel .env locale per run_aws.sh / run_test_engine_ecs.sh."
  # MODIFICA: Punta alla variabile locale anziché al data source
  value       = local.datasets_bucket_name
}

output "security_group_id" {
  description = "ID del Security Group condiviso da orchestrator/worker/test-engine."
  value       = aws_security_group.rf_distributed.id
}

output "training_mode" {
  description = "Modalità di training con cui è stata deployata l'infrastruttura."
  value       = var.training_mode
}

output "api_gateway_endpoint" {
  description = "URL base dell'API Gateway. Da usare come API_GATEWAY_URL nel .env locale."
  value       = aws_apigatewayv2_api.mljobs.api_endpoint
}

output "worker_service_names" {
  description = "Nomi dei Service worker creati (dipende dalla modalità)."
  value = var.training_mode == "federated" ? [
    for s in aws_ecs_service.worker_federated : s.name
    ] : [
    aws_ecs_service.worker_centralized[0].name
  ]
}

output "next_steps" {
  description = "Comandi utili dopo il primo apply."
  value = <<-EOT
    Infrastruttura creata. Prossimi passi:

    1. Aggiorna il tuo .env locale con:
         SYS_ENV=aws
         SYS_MODE=${var.training_mode}
         DATASETS_BUCKET_NAME=${local.datasets_bucket_name}
         AWS_DEFAULT_REGION=${var.aws_region}
         NUM_WORKERS=${var.num_workers}
         API_GATEWAY_URL=${aws_apigatewayv2_api.mljobs.api_endpoint}

    2. Se training_mode=federated e non l'hai già fatto, esegui il provisioning
       degli shard PRIMA di sottomettere un job:
         python -m scripts.provision_federated_shards --num-workers ${var.num_workers}

    3. Avvia il client contro l'infrastruttura:
         ./run_aws.sh

    4. Per fermare tutto senza distruggere l'infrastruttura (scala i Service a 0):
         aws ecs update-service --cluster ${var.cluster_name} --service orchestrator-service --desired-count 0 --region ${var.aws_region}
         (ripeti per ciascun worker-service)

    5. Per distruggere TUTTA l'infrastruttura creata da Terraform:
         terraform destroy
  EOT
}
