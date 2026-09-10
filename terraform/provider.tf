provider "aws" {
  region = var.aws_region
  
  # Indica a Terraform di usare lo stile standard per gli endpoint S3
  s3_use_path_style = true
  # DISATTIVA I CONTROLLI CHE FANNO FALLIRE IL LAB:
  # Dice al provider di non chiamare le API di validazione avanzata dei metadati,
  # delle credenziali e delle regioni, che spesso invocano policy protette.
  skip_metadata_api_check     = true
  skip_region_validation      = true
  skip_credentials_validation = true

  # La SCP del Learner Lab nega la creazione di risorse ECS (task
  # definition, cluster, ecc.) se la richiesta non porta almeno un tag.
  # default_tags applica questo tag automaticamente a ogni risorsa AWS
  # creata da Terraform che supporta il tagging, senza doverlo scrivere
  # a mano su ogni resource block.
  #
  # ATTENZIONE: il valore DEVE combaciare con var.project_name
  # ("rf-distributed", vedi variables.tf) - è lo stesso usato in ogni
  # filtro "tag:Project" del progetto (run_test_engine.sh,
  # orchestrator_fault.py, orchestrator_asg_replacement.py, README). Un
  # valore diverso qui non rompe la creazione delle risorse via Terraform,
  # ma le rende invisibili a quei filtri.
  default_tags {
    tags = {
      Project = "rf-distributed"
    }
  }
}