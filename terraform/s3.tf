data "aws_s3_bucket" "datasets" {
  bucket = "${var.project_name}-datasets-${data.aws_caller_identity.current.account_id}-${var.aws_region}"
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = data.aws_s3_bucket.datasets.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = data.aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}