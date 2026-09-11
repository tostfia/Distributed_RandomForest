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
    Infrastruttura creata (modalità: ${var.training_mode}). Prossimi passi:

    1. Aggiorna il tuo .env locale con:
         SYS_ENV=aws
         SYS_MODE=${var.training_mode}
         DATASETS_BUCKET_NAME=${local.datasets_bucket_name}
         AWS_DEFAULT_REGION=${var.aws_region}
         NUM_WORKERS=${var.num_workers}
         API_GATEWAY_URL=${aws_apigatewayv2_api.mljobs.api_endpoint}

    2. Provisioning shard: SOLO per dataset_type=real (partizionamento
       by_day su S3). Per dataset_type=synthetic NON va eseguito: i dati
       vengono generati al volo al primo training.
    %{ if var.training_mode == "federated" }
         python -m scripts.provision_federated_shards --num-workers ${var.num_workers}
    %{ endif }

    3. AVVIA l'infrastruttura (parte FERMA dopo l'apply: worker_desired_count
       e orchestrator_desired_count di default sono 0, nessun task/istanza
       in esecuzione finché non li alzi):

         aws autoscaling update-auto-scaling-group --auto-scaling-group-name orchestrator-asg \
           --min-size 2 --max-size 2 --desired-capacity 2 --region ${var.aws_region}

    %{ if var.training_mode == "federated" }
         for i in $(seq 1 ${var.num_workers}); do
           aws ecs update-service --cluster ${var.cluster_name} --service "worker-service-$i" \
             --desired-count 1 --region ${var.aws_region} > /dev/null
         done
    %{ else }
         aws ecs update-service --cluster ${var.cluster_name} --service worker-service \
           --desired-count ${var.num_workers} --region ${var.aws_region}
    %{ endif }

       Aspetta 3-4 minuti (boot EC2, pull immagine, avvio container) prima
       di procedere - vedi README, sezione 6.1, per come verificarlo.

    4. Avvia il client contro l'infrastruttura:
         ./run_aws.sh

    5. Per fermare tutto senza distruggere l'infrastruttura (stessi comandi
       del punto 3, con --desired-capacity 0 / --desired-count 0):

         aws autoscaling update-auto-scaling-group --auto-scaling-group-name orchestrator-asg \
           --min-size 0 --max-size 0 --desired-capacity 0 --region ${var.aws_region}
    %{ if var.training_mode == "federated" }
         for i in $(seq 1 ${var.num_workers}); do
           aws ecs update-service --cluster ${var.cluster_name} --service "worker-service-$i" \
             --desired-count 0 --region ${var.aws_region} > /dev/null
         done
    %{ else }
         aws ecs update-service --cluster ${var.cluster_name} --service worker-service \
           --desired-count 0 --region ${var.aws_region}
    %{ endif }

       Per rendere lo stop persistente attraverso un futuro apply, imposta
       in terraform.tfvars: orchestrator_desired_count = 0, worker_desired_count = 0
       (sono già i default se non li specifichi affatto).

    6. Per distruggere TUTTA l'infrastruttura creata da Terraform:
         terraform destroy
  EOT
}