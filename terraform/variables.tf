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
  default     = 3
}

variable "orchestrator_desired_count" {
  description = "Numero di istanze orchestrator (>=2 per testare la leader election / failover)."
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

variable "orchestrator_cpu" {
  description = "vCPU allocate per orchestrator in unità Fargate."
  type        = string
  default     = "2048"
}

variable "orchestrator_memory" {
  description = "Memoria (MiB) allocata per orchestrator."
  type        = string
  default     = "8192"
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
