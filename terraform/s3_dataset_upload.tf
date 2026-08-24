# ---------------------------------------------------------------------
# Upload automatico del dataset reale locale (dataset_cache/) su S3,
# sotto il prefisso 'real/' — lo stesso path che il client propone come
# default quando si sceglie "Dataset REALE" nel menù interattivo
# (s3://<bucket>/real/).
#
# for_each su fileset(): ogni file locale diventa un aws_s3_object.
# 'etag = filemd5(...)' fa sì che Terraform ri-carichi un file SOLO se
# il suo contenuto è cambiato dall'ultimo apply, non ad ogni deploy.
#
# NOTA: presuppone che 'dataset_cache/' esista alla root del progetto
# (var.source_path, di default '..' rispetto a terraform/), allo stesso
# livello del Dockerfile. Se il dataset è altrove, aggiorna il percorso
# nel fileset() sotto.
# ---------------------------------------------------------------------

resource "aws_s3_object" "real_dataset" {
  for_each = fileset("${var.source_path}/dataset_cache", "**")

  bucket = aws_s3_bucket.datasets.id
  key    = "real/${each.value}"
  source = "${var.source_path}/dataset_cache/${each.value}"
  etag   = filemd5("${var.source_path}/dataset_cache/${each.value}")

  tags = { Project = var.project_name }
}
