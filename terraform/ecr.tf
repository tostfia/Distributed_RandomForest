resource "aws_ecr_repository" "rf_distributed" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE" # coerente con l'uso del tag 'latest' riscritto ad ogni deploy

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = var.project_name
  }
}
