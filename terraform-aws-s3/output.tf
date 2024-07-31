output "s3_bucket_id" {
  value       = data.aws_s3_bucket.customer_segmentation_data.id
  description = "The ID of the S3 bucket"
}

output "s3_bucket_arn" {
  value       = data.aws_s3_bucket.customer_segmentation_data.arn
  description = "The ARN of the S3 bucket"
}

output "s3_bucket_region" {
  value       = data.aws_s3_bucket.customer_segmentation_data.region
  description = "The region where the S3 bucket is created"
}