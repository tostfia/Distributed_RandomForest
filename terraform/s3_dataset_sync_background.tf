# ---------------------------------------------------------------------
# Sync automatico del dataset reale locale su S3, lanciato in background
# (nohup + disown) durante 'terraform apply'.
#
# Sostituisce l'approccio precedente basato su aws_s3_object per-file:
# quello richiedeva che 'terraform apply' restasse aperto per tutta la
# durata dell'upload (problema se la macchina va in sospensione o il
# terminale si chiude); questo invece lancia 'aws s3 sync' come processo
# indipendente e torna subito, quindi l'apply si completa in pochi
# secondi mentre il trasferimento prosegue per conto suo.
#
# Il trigger è un hash di tutti i file in dataset_cache/: il sync
# riparte SOLO se qualcosa è cambiato dall'ultimo apply, non ad ogni
# esecuzione.
#
# NOTA: essendo async, terraform apply NON aspetta il completamento
# dell'upload. Controlla l'avanzamento con:
#   tail -f ../upload_dataset.log
# ---------------------------------------------------------------------

resource "null_resource" "dataset_sync" {
  triggers = {
    dataset_hash = sha1(join("", [
      for f in sort(fileset("${var.source_path}/dataset_cache", "**")) :
      filemd5("${var.source_path}/dataset_cache/${f}")
    ]))
  }

  provisioner "local-exec" {
    command = <<-EOT
      nohup aws s3 sync ${var.source_path}/dataset_cache s3://${aws_s3_bucket.datasets.bucket}/real/ --region ${var.aws_region} > ${var.source_path}/upload_dataset.log 2>&1 &
      disown
      echo "Sync dataset avviato in background (log: upload_dataset.log). Verifica con: tail -f ${var.source_path}/upload_dataset.log"
    EOT
  }

  depends_on = [aws_s3_bucket.datasets]
}
