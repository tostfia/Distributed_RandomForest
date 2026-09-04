# ---------------------------------------------------------------------
# Configurazione S3 - Ottimizzata per limiti AWS Academy / Vocareum
#
# NOTA PER LA VALUTAZIONE: Il provider AWS di Terraform tenta implicitamente
# di leggere la configurazione di Object Lock (GetBucketObjectLockConfiguration) 
# sia sulle risorse che sui Data Source di S3. Nelle sandbox AWS Academy questa API 
# è bloccata da una Service Control Policy (SCP) centralizzata. 
# Per aggirare definitivamente il problema mantenendo l'automazione, il nome 
# del bucket viene calcolato unicamente in locale come stringa pura, mentre il bucket 
# viene creato a monte (via console o CLI, vedi README.md).
# ---------------------------------------------------------------------

locals {
  # Calcoliamo il nome del bucket in locale per non interpellare le API S3 di AWS
  datasets_bucket_name = "${var.project_name}-datasets-${data.aws_caller_identity.current.account_id}-${var.aws_region}"
}
