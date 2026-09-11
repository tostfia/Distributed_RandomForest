variable "aws_region" {
  description = "Regione AWS in cui creare tutte le risorse."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefisso usato per nominare le risorse globalmente uniche (bucket S3, repo ECR)."
  type        = string
  default     = "rf-distributed"
}

variable "cluster_name" {
  description = "Nome del cluster ECS."
  type        = string
  default     = "forest-cluster"
}

variable "training_mode" {
  description = "Modalità di training: 'centralized' (worker anonimi intercambiabili) o 'federated' (worker con indice fisso, uno per shard)."
  type        = string
  default     = "centralized"
  validation {
    condition     = contains(["centralized", "federated"], var.training_mode)
    error_message = "training_mode deve essere 'centralized' o 'federated'."
  }
}

variable "num_workers" {
  description = "Numero di worker da avviare (desired-count in centralized, numero di indici fissi 1..N in federated)."
  type        = number
  default     = 10
}

variable "worker_desired_count" {
  description = "Quanti worker sono EFFETTIVAMENTE avviati (desired_count reale), disaccoppiato da num_workers. Default 0: l'apply crea tutte le risorse (task definition, N service in federated) ma le lascia ferme - nessun task Fargate parte, nessun costo di calcolo, finché non alzi questo valore (via un nuovo apply, o direttamente con 'aws ecs update-service --desired-count' sui service già creati, come già fai oggi). In centralized, un valore diverso da 0 o num_workers permette anche un avvio parziale (es. 5 worker attivi su 10 provisionati). In federated ogni service ha comunque al massimo 1 task (un worker per indice/shard): qualunque valore > 0 qui porta TUTTI i service federated a desired_count=1, 0 li lascia tutti fermi - non esiste un avvio parziale per indice via questa variabile (per quello, resta il comando aws ecs update-service mirato su un singolo worker-service-N)."
  type        = number
  default     = 0
}

variable "orchestrator_desired_count" {
  description = "Numero di istanze EC2 dell'orchestrator (>=2 per testare la leader election / failover). Non più un desired-count ECS: l'orchestrator gira su istanze EC2 dedicate, vedi orchestrator_ec2.tf (la SCP del Learner Lab nega task definition ECS con memoria > 8192 MiB, insufficiente per gli scenari di scalabilità pesanti)."
  type        = number
  default     = 2
}

variable "worker_cpu" {
  description = "vCPU allocate per worker in unità Fargate (1024 = 1 vCPU). NOTA: 4096 (4 vCPU) è il default storico del progetto per permettere parallelismo reale nel pool di processi; abbassalo (es. 2048) durante lo sviluppo quotidiano per ridurre i costi."
  type        = string
  default     = "4096"
}

variable "worker_memory" {
  description = "Memoria (MiB) allocata per worker. Deve essere un valore compatibile con worker_cpu secondo le combinazioni Fargate."
  type        = string
  default     = "8192"
}

variable "test_engine_cpu" {
  type    = string
  default = "2048"
}

variable "test_engine_memory" {
  type    = string
  default = "16384"
}

variable "rpc_port" {
  description = "Porta RPC (rpyc) usata da orchestrator per parlare con i worker."
  type        = number
  default     = 18861
}

variable "rpc_sync_timeout_seconds" {
  description = "Timeout (secondi) delle chiamate RPC sincrone dell'orchestrator verso i worker durante il training."
  type        = string
  default     = "1800"
}

variable "rpc_inference_sync_timeout_seconds" {
  description = "Timeout (secondi) delle chiamate RPC sincrone dell'orchestrator verso i worker durante l'inferenza."
  type        = string
  default     = "900"
}

variable "worker_heartbeat_timeout" {
  description = "Timeout (secondi) di heartbeat oltre il quale un worker è considerato morto."
  type        = string
  default     = "120"
}

variable "image_tag" {
  description = "Tag dell'immagine Docker da buildare e usare nei task ECS."
  type        = string
  default     = "latest"
}

variable "force_image_rebuild" {
  description = "Se true, forza sempre il rebuild+push dell'immagine Docker ad ogni apply, anche senza modifiche al codice sorgente. Utile in CI, sconsigliato per iterazione locale rapida (rallenta ogni apply)."
  type        = bool
  default     = false
}

variable "source_path" {
  description = "Percorso della root del progetto (dove sta il Dockerfile), relativo a questa cartella terraform/."
  type        = string
  default     = ".."
}