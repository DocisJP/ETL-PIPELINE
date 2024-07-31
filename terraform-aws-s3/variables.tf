variable "aws_region" {
  description = "The AWS region to create resources in"
  default     = "sa-east-1"
}

variable "bucket_name" {
  description = "The name of the S3 bucket"
  default     = "jp-customer-segmentation-data-20240727"
}

variable "force_destroy" {
  description = "A boolean that indicates all objects should be deleted from the bucket so that the bucket can be destroyed without error"
  type        = bool
  default     = true
}

variable "enable_versioning" {
  description = "A boolean to enable/disable versioning on the bucket"
  type        = bool
  default     = true
}