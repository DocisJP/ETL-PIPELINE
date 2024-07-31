# ETL Pipeline Documentation

## 1. ETL Pipeline Overview (etl_pipeline.py)

The ETL (Extract, Transform, Load) pipeline is the core of this data processing system. It's responsible for extracting data from an S3 bucket, transforming it through various data cleaning and enrichment steps, and preparing it for loading into Snowflake.

### Key Functions:

a) `load_config(config_path: str) -> Dict[str, Any]`:
   - Loads configuration from a YAML file.
   - Used to centralize configuration management.

b) `extract_data_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame`:
   - Extracts data from an S3 bucket using boto3.
   - Returns the data as a pandas DataFrame.

c) `clean_date_format(date_str)`:
   - Attempts to clean and standardize date formats.
   - Returns a standardized date string or NaN if parsing fails.

d) `encode_categorical_columns(df)`:
   - Converts specified columns to categorical data type.
   - Improves memory efficiency and performance for categorical data.

e) `standardize_categorical_formats(df)`:
   - Applies title case to categorical columns.
   - Standardizes education level mappings.
   - Capitalizes policy-related columns.

f) `handle_outliers_vectorized(df, columns)`:
   - Uses a vectorized approach to handle outliers in specified columns.
   - Applies interquartile range (IQR) method for outlier detection and capping.

g) `impute_missing_values(df)`:
   - Uses KNNImputer to fill missing values in numeric columns.

h) `perform_kmeans_clustering(df)`:
   - Applies K-means clustering on scaled numeric features.
   - Adds a 'Cluster' column to the DataFrame.

i) `perform_pca(df, scaled_features)`:
   - Performs Principal Component Analysis (PCA) on scaled features.
   - Adds 'PCA1' and 'PCA2' columns to the DataFrame.

j) `calculate_risk_score(df)`:
   - Calculates a risk score based on Age, Income Level, and Coverage Amount.
   - Normalizes the risk score between 0 and 1.

k) `transform_data(df)`:
   - Orchestrates the entire transformation process.
   - Applies all the above functions in sequence.

l) `etl_process(config: Dict[str, Any]) -> pd.DataFrame`:
   - The main ETL function that orchestrates the entire process.
   - Extracts data, applies transformations, and prepares it for loading.
   - Includes extensive logging and validation steps.

## 2. Data Versioning (data_versioning.py)

This module handles different versions of the data format, allowing the ETL process to adapt to changes in the input data structure.

### Key Functions:

a) `check_data_format_version(df: pd.DataFrame) -> int`:
   - Determines the version of the input data based on its structure.
   - Caches the version to avoid repeated checks.

b) `reset_version_cache()`:
   - Resets the cached version, useful when starting a new ETL process.

c) `version_handler(func: Callable) -> Callable`:
   - A decorator to make functions version-aware.
   - Passes the detected version to the decorated function.

d) `parse_date(date_string)`:
   - Attempts to parse dates in various formats.
   - Returns a pandas datetime object or NaT if parsing fails.

e) `preprocess_data(df: pd.DataFrame, version: int = 1) -> pd.DataFrame`:
   - Applies version-specific preprocessing steps.
   - Handles date parsing, numeric column processing, and categorical encoding.

f) `add_time_based_features(df: pd.DataFrame, version: int = 1) -> pd.DataFrame`:
   - Adds time-based features like day of week, month, season, and customer tenure.
   - Includes version-specific processing (e.g., adding Purchase_Year for version 2).

## 3. Performance Monitoring (performance.py)

This module provides tools for monitoring the performance of the ETL process.

### Key Functions:

a) `performance_monitor(func)`:
   - A decorator that measures execution time and memory usage of functions.

b) `log_overall_performance(start_time, start_memory)`:
   - Logs the overall execution time and memory usage of the ETL process.

c) `get_current_memory_usage()`:
   - Returns the current memory usage of the process.

## 4. Validation (validation.py)

This module contains functions for validating the data at various stages of the ETL process.

### Key Functions:

a) `validate_date_format(df)`:
   - Checks if the 'Purchase_History' column has a valid date format.

b) `validate_data(df, log_results=True)`:
   - Performs comprehensive data validation checks.
   - Checks for missing values, data types, numeric ranges, and unique values in categorical columns.

c) `log_dataframe_info(df: pd.DataFrame, stage: str)`:
   - Logs basic information about the DataFrame at different stages of the ETL process.

d) `log_detailed_dataframe_info(df: pd.DataFrame, stage: str, num_rows: int = 10)`:
   - Logs detailed information about the DataFrame, including sample data, column types, and statistics.

## 5. Snowflake Loader (snowflake_loader.py)

This module handles the loading of transformed data into Snowflake.

### Key Functions:

a) `load_config(config_path)`:
   - Loads Snowflake configuration from a YAML file.

b) `load_data_to_snowflake(df, config)`:
   - Establishes a connection to Snowflake using the provided configuration.
   - Uses snowflake.connector.pandas_tools.write_pandas to efficiently write the DataFrame to Snowflake.

## 6. Infrastructure (create.sql)

This SQL script sets up the necessary Snowflake infrastructure:
- Creates a warehouse, database, and schema.
- Defines the structure of the customer_segmentation table.
- Creates a materialized view for customer risk summary.
- Sets up clustering for improved query performance.

## 7. Testing (test_etl.py)

This module contains unit tests for the ETL pipeline, ensuring the correctness of various transformation steps and the overall process.

## 8. Requirements (requirements.txt)

Lists all the Python packages required to run the ETL pipeline, ensuring reproducibility and easy setup.

## Summary

In summary, this ETL pipeline is a robust, version-aware system that extracts data from S3, applies a series of transformations to clean and enrich the data, and prepares it for loading into Snowflake. It includes performance monitoring, extensive data validation, and is designed to handle different versions of the input data format. The system is well-documented, includes unit tests, and is set up for easy deployment and scaling in a cloud environment.