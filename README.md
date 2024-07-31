# Data Engineering ETL Pipeline Project

## Introduction

This project implements an end-to-end ETL (Extract, Transform, Load) pipeline for processing customer segmentation data. It demonstrates modern data engineering practices, including:

- Data extraction from AWS S3
- Data transformation and analysis using Python
- Data loading into Snowflake data warehouse
- Infrastructure as Code using Terraform
- Comprehensive data validation and versioning
- Automated testing

The pipeline handles customer data, performs various transformations including feature engineering and machine learning techniques, and prepares the data for analysis in Snowflake.
For more deatail you can go [here](https://github.com/DocisJP/ETL-PIPELINE/blob/main/santex-challenge-etl/PythonReadme.md).

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup and Dependencies](#setup-and-dependencies)
   - [System Requirements](#system-requirements)
   - [Initial Setup](#initial-setup)
   - [AWS Setup](#aws-setup)
   - [Terraform Setup](#terraform-setup)
3. [ETL Pipeline (etl_pipeline.py)](#etl-pipeline-etl_pipelinepy)
   - [Data Extraction](#data-extraction)
   - [Data Transformation](#data-transformation)
   - [Main ETL Process](#main-etl-process)
4. [Data Validation (validation.py)](#data-validation-validationpy)
5. [Data Versioning (data_versioning.py)](#data-versioning-data_versioningpy)
6. [Testing (test_etl.py)](#testing-test_etlpy)
7. [Snowflake Setup (create.sql)](#snowflake-setup-createsql)
8. [Snowflake Data Loading (snowflake_loader.py)](#snowflake-data-loading-snowflake_loaderpy)
9. [Validation Images](#validation-images)
10. [Usage](#usage)

## Project Structure

```
santex-challenge-etl/
│
├── etl_pipeline.py
├── validation.py
├── data_versioning.py
├── test_etl.py
├── snowflake_loader.py
├── performance.py
├── requirements.txt
├── config.yml
└── README.md
```

## Setup and Dependencies

### System Requirements

- Ubuntu OS
- Python 3.8+
- AWS CLI
- Terraform 1.0+
- Snowflake account

### Initial Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DocisJP/ETL-PIPELINE.git
   cd data-engineering-etl-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up AWS CLI (details in AWS Setup section)

5. Install and configure Terraform (details in Terraform Setup section)

6. Update `config.yml` with your AWS and Snowflake credentials

### AWS Setup

AWS is used for storing the raw data and as the source for our ETL pipeline. Here's how to set it up:
In this case I also creatd a new user in the IAM roles, giving it acces to manipulate the CLI and the buckets. 
It's a good practice and I recommend following up on it.

1. Install AWS CLI:
   ```bash
   sudo apt-get update
   sudo apt-get install awscli
   ```

2. Configure AWS CLI with a named profile:
   ```bash
   aws configure --profile data-challenge
   ```
   Enter your AWS Access Key ID, Secret Access Key, default region (e.g., sa-east-1), and output format (json).

3. Set the AWS profile for your current session:
   ```bash
   export AWS_PROFILE=data-challenge
   ```

4. Verify the configuration:
   ```bash
   aws configure list
   ```

### Terraform Setup

Terraform is used to manage our AWS infrastructure as code. Here's how to set it up and what it does in this project:

1. Install Terraform:
   ```bash
   curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
   sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
   sudo apt-get update && sudo apt-get install terraform
   ```

2. Initialize Terraform in the project directory:
   ```bash
   terraform init
   ```

3. Review and apply the configuration:
   ```bash
   terraform plan
   terraform apply
   ```

4. Confirm by typing `yes` when prompted

After applying, you'll have an S3 bucket set up for your project data.

### Data Upload

To upload the initial dataset to S3:
```bash
aws s3 cp customer_segmentation_data.csv s3://your-bucket
```

Verify the upload:
```bash
aws s3 ls s3://your-data/
```

## ETL Pipeline (etl_pipeline.py)

The `etl_pipeline.py` file contains the main ETL process. 

### Data Extraction

- Function: `extract_data_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame`
- Description: Extracts data from an AWS S3 bucket and returns it as a pandas DataFrame.

### Data Transformation

- Function: `transform_data(df: pd.DataFrame) -> pd.DataFrame`
- Description: Applies a series of transformations to the input DataFrame, including:
  - Preprocessing (handling dates, encoding categorical variables)
  - Adding time-based features
  - Handling outliers
  - Imputing missing values
  - Performing K-means clustering
  - Applying PCA (Principal Component Analysis)
  - Calculating risk scores

### Main ETL Process

- Function: `etl_process(config: Dict[str, Any]) -> pd.DataFrame`
- Description: Orchestrates the entire ETL process, including:
  - Data extraction from S3
  - Data transformation
  - Data validation
  - Performance monitoring

## Data Validation (validation.py)

The `validation.py` file contains functions for validating the data throughout the ETL process:

### Key Functions:

1. `validate_date_format(df: pd.DataFrame) -> bool`
   - Validates the format of dates in the 'Purchase_History' column.

2. `validate_data(df: pd.DataFrame, log_results: bool = True) -> Dict`
   - Performs comprehensive data validation, checking for:
     - Missing values
     - Data types
     - Numeric ranges
     - Unique values in categorical columns

3. `log_dataframe_info(df: pd.DataFrame, stage: str)`
   - Logs basic information about the DataFrame at different stages of the ETL process.

4. `log_detailed_dataframe_info(df: pd.DataFrame, stage: str, num_rows: int = 10)`
   - Logs detailed information about the DataFrame, including sample data, column types, and basic statistics.

## Data Versioning (data_versioning.py)

The `data_versioning.py` file handles different versions of the data schema:

### Key Components:

1. `check_data_format_version(df: pd.DataFrame) -> int`
   - Determines the version of the input data based on its structure.

2. `version_handler(func: Callable) -> Callable`
   - Decorator to make functions version-aware.

3. `preprocess_data(df: pd.DataFrame, version: int = 1) -> pd.DataFrame`
   - Preprocesses the data based on its version.

4. `add_time_based_features(df: pd.DataFrame, version: int = 1) -> pd.DataFrame`
   - Adds time-based features to the DataFrame, with version-specific processing.

## Testing (test_etl.py)

The `test_etl.py` file contains unit tests for the ETL pipeline:

### Test Cases:

1. `test_data_validation`
2. `test_data_format_version_detection`
3. `test_preprocess_data`
4. `test_add_time_based_features`
5. `test_transformed_data_validation`
6. `test_encode_categorical_columns`
7. `test_standardize_categorical_formats`
8. `test_handle_outliers_vectorized`
9. `test_impute_missing_values`
10. `test_perform_kmeans_clustering`
11. `test_perform_pca`
12. `test_calculate_risk_score`
13. `test_end_to_end_transformation`

## Snowflake Setup (create.sql)

The `create.sql` file contains the SQL commands to set up the necessary Snowflake objects for this project.

### Warehouse Creation

```sql
CREATE WAREHOUSE IF NOT EXISTS customer_segmentation_wh
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE;
```

### Database and Schema Creation

```sql
CREATE DATABASE IF NOT EXISTS customer_segmentation_db;
USE DATABASE customer_segmentation_db;
CREATE SCHEMA IF NOT EXISTS etl_schema;
USE SCHEMA etl_schema;
```

### Table Creation

```sql
CREATE OR REPLACE TABLE customer_segmentation (
    Customer_ID INT,
    Age INT,
    Gender VARCHAR(10),
    -- ... other columns ...
    Risk_Score FLOAT
);
```

### Optimization

```sql
ALTER TABLE customer_segmentation CLUSTER BY (segmentation_group, cluster);
```

### Materialized View

```sql
CREATE OR REPLACE MATERIALIZED VIEW customer_risk_summary AS
SELECT segmentation_group, cluster, AVG(risk_score) as avg_risk_score
FROM customer_segmentation
GROUP BY segmentation_group, cluster;
```

## Snowflake Data Loading (snowflake_loader.py)

The `snowflake_loader.py` file handles the process of loading the transformed data into Snowflake:

### Key Components:

1. `load_config(config_path: str) -> Dict`
   - Loads the configuration from the YAML file.

2. `load_data_to_snowflake(df: pd.DataFrame, config: Dict)`
   - Establishes a connection to

 Snowflake and loads the DataFrame into the specified table.

3. `main(config_path: str)`
   - Orchestrates the data loading process, including validation and logging.

### Snowflake Account Setup

1. Set up a Snowflake account (if you don't have one)
2. Create a Snowflake user and role with appropriate permissions

## Performance Monitoring (performance.py)

The `performance.py` file monitors the ETL pipeline's performance.

### Key Metrics:

1. `@log_execution_time`
   - Decorator to log the execution time of functions.

2. `monitor_memory_usage(func: Callable) -> Callable`
   - Decorator to monitor memory usage during the ETL process.

## Validation Images

Validation images can be found in the `validation_images` folder, providing visual confirmation of successful validation steps.

## Usage

1. Update `config.yml` with your AWS and Snowflake credentials.
2. Run the ETL pipeline:
   ```bash
   python etl_pipeline.py
   ```
3. Load data into Snowflake:
   ```bash
   python snowflake_loader.py
   ```

### Example Commands

#### Run ETL Pipeline

```bash
python etl_pipeline.py
```

#### Load Data into Snowflake

```bash
python snowflake_loader.py
```

