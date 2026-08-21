# ---------------------------------------------------------------------
# Build + push dell'immagine Docker, equivalente ai passi [5/10] e [6/10]
# di deploy.sh, ma eseguiti da Terraform stesso con un provisioner
# local-exec: chi lancia 'terraform apply' deve avere Docker installato
# e in esecuzione sulla propria macchina (esattamente come per deploy.sh).
#
# I 'triggers' fanno sì che il rebuild avvenga SOLO se cambia qualcosa di
# rilevante (Dockerfile o codice sorgente in src/), non ad ogni apply:
# altrimenti ogni piccola modifica infrastrutturale (es. numero worker)
# forzerebbe un rebuild+push completo, inutile e lento.
#
# force_image_rebuild=true bypassa questa cache e forza sempre il rebuild
# (utile se il meccanismo di hashing non rileva una dipendenza cambiata,
# es. un file fuori da src/ referenziato dal Dockerfile).
# ---------------------------------------------------------------------
resource "null_resource" "docker_build_push" {
  triggers = {
    dockerfile_hash = filemd5("${var.source_path}/Dockerfile")
    src_hash        = sha1(join("", [for f in sort(fileset(var.source_path, "src/**")) : filemd5("${var.source_path}/${f}")]))
    force_rebuild    = var.force_image_rebuild ? timestamp() : "static"
  }

  provisioner "local-exec" {
    working_dir = var.source_path
    command     = <<-EOT
      set -e
      echo "==> Login Docker su ECR..."
      aws ecr get-login-password --region ${var.aws_region} | \
        docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com

      echo "==> Build immagine Docker..."
      docker build -t ${var.project_name} .

      echo "==> Tag e push su ECR..."
      docker tag ${var.project_name}:latest ${aws_ecr_repository.rf_distributed.repository_url}:${var.image_tag}
      docker push ${aws_ecr_repository.rf_distributed.repository_url}:${var.image_tag}
    EOT
  }

  depends_on = [aws_ecr_repository.rf_distributed]
}
