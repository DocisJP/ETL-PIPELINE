# Use existing S3 bucket
data "aws_s3_bucket" "customer_segmentation_data" {
  bucket = var.bucket_name
}

# Enable versioning for the S3 bucket
resource "aws_s3_bucket_versioning" "customer_data_versioning" {
  bucket = data.aws_s3_bucket.customer_segmentation_data.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

# Configure server-side encryption for the S3 bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "customer_data_encryption" {
  bucket = data.aws_s3_bucket.customer_segmentation_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "customer_data_public_access_block" {
  bucket = data.aws_s3_bucket.customer_segmentation_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}