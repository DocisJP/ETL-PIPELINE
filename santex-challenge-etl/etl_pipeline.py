import boto3
import pandas as pd
import numpy as np
from io import StringIO
from functools import reduce
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import time
from performance import performance_monitor, log_overall_performance, get_current_memory_usage
import logging
import yaml
from typing import Dict, Any
from validation import validate_date_format, validate_data, log_dataframe_info, log_detailed_dataframe_info
from data_versioning import preprocess_data, add_time_based_features, check_data_format_version, reset_version_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def extract_data_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    try:
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        return pd.read_csv(StringIO(content))
    except Exception as e:
        logger.error(f"Error extracting data from S3: {e}")
        raise

def clean_date_format(date_str):
    try:
        return pd.to_datetime(date_str, format='%d-%m-%Y').strftime('%d-%m-%Y')
    except:
        return np.nan


@performance_monitor
def encode_categorical_columns(df):
    categorical_columns = ['Gender', 'Marital_Status', 'Education_Level', 'Geographic_Information', 
                           'Occupation', 'Behavioral_Data', 'Interactions_with_Customer_Service', 
                           'Insurance_Products_Owned', 'Policy_Type', 'Customer_Preferences', 
                           'Preferred_Communication_Channel', 'Preferred_Contact_Time', 
                           'Preferred_Language', 'Segmentation_Group', 'Age_Group', 'Purchase_Season']
    
    existing_cat_columns = [col for col in categorical_columns if col in df.columns]
    
    df[existing_cat_columns] = df[existing_cat_columns].astype('category')
    
    return df


def standardize_categorical_formats(df):
    # Define categorical columns
    categorical_columns = ['Gender', 'Marital_Status', 'Education_Level', 'Occupation', 'Policy_Type', 
                           'Preferred_Communication_Channel', 'Preferred_Language', 'Segmentation_Group', 
                           'Behavioral_Data', 'Insurance_Products_Owned']
    
    # Filter existing columns
    existing_columns = df.columns.intersection(categorical_columns)
    
    # Apply title case to all existing categorical columns at once
    df[existing_columns] = df[existing_columns].astype(str).apply(lambda x: x.str.title())
    
    # Education level mapping
    education_mapping = {
        "Bachelor'S Degree": "Bachelor's Degree",
        "Master'S Degree": "Master's Degree",
        "Associate Degree": "Associate's Degree",
        "Associate'S Degree": "Associate's Degree"
    }
    
    # Apply education mapping if the column exists
    if 'Education_Level' in df.columns:
        df['Education_Level'] = df['Education_Level'].replace(education_mapping)
    
    # Policy columns
    policy_columns = ['Behavioral_Data', 'Insurance_Products_Owned']
    existing_policy_columns = df.columns.intersection(policy_columns)
    
    # Capitalize policy columns
    df[existing_policy_columns] = df[existing_policy_columns].apply(lambda x: x.str.capitalize())
    
    return df

def handle_outliers_vectorized(df, columns):
    columns = [col.replace(' ', '_') for col in columns]
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(df[columns].values, lower_bound, upper_bound)

def impute_missing_values(df):
    numeric_columns = ['Income_Level', 'Coverage_Amount', 'Premium_Amount']
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def perform_kmeans_clustering(df):
    numeric_columns = ['Income_Level', 'Coverage_Amount', 'Premium_Amount']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numeric_columns])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features).astype('int64')
    return df, scaled_features

def perform_pca(df, scaled_features):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df[['PCA1', 'PCA2']] = pca_result
    return df

def calculate_risk_score(df):
    df['Risk_Score'] = (df['Age'].values * 0.3 + (1 - df['Income_Level'].values) * 0.4 + df['Coverage_Amount'].values * 0.3) / 3
    df['Risk_Score'] = (df['Risk_Score'] - df['Risk_Score'].min()) / (df['Risk_Score'].max() - df['Risk_Score'].min())
    return df

@performance_monitor
def transform_data(df):
    transformations = [
        preprocess_data,
        add_time_based_features,
        lambda df: df.rename(columns=lambda x: x.replace(' ', '_')),
        standardize_categorical_formats,
        lambda df: df.assign(**dict(zip(['Income_Level', 'Coverage_Amount', 'Premium_Amount'], 
                                        handle_outliers_vectorized(df, ['Income_Level', 'Coverage_Amount', 'Premium_Amount']).T))),
        impute_missing_values,
        perform_kmeans_clustering,
        lambda df_scaled: perform_pca(*df_scaled),
        calculate_risk_score,
        encode_categorical_columns,
        lambda df: df.astype({col: 'float32' for col in df.select_dtypes(include=['float64']).columns})
    ]
    return reduce(lambda df, func: func(df), transformations, df)

@performance_monitor
def etl_process(config: Dict[str, Any]) -> pd.DataFrame:
    bucket_name = config['s3']['bucket_name']
    file_key = config['s3']['file_key']

    try:
        start_time = time.time()
        start_memory = get_current_memory_usage()

        reset_version_cache()

        with ThreadPoolExecutor() as executor:
            future = executor.submit(extract_data_from_s3, bucket_name, file_key)
            raw_data = future.result()

        log_dataframe_info(raw_data, "after extraction")

        version = check_data_format_version(raw_data)
        logger.info(f"Detected data format version: {version}")
        logger.info(f"Number of rows in raw data: {len(raw_data)}")

        raw_validation = validate_data(raw_data)
        
        if not validate_date_format(raw_data):
            logger.warning("Date format validation failed in raw data. Proceeding with transformation.")

        try:
            preprocessed_data = preprocess_data(raw_data)
            log_detailed_dataframe_info(preprocessed_data, "after preprocessing")

            time_featured_data = add_time_based_features(preprocessed_data)
            log_detailed_dataframe_info(time_featured_data, "after adding time-based features")

            standardized_data = standardize_categorical_formats(time_featured_data)
            log_detailed_dataframe_info(standardized_data, "after standardizing categorical formats")

            outlier_handled_data = standardized_data.assign(**dict(zip(['Income_Level', 'Coverage_Amount', 'Premium_Amount'], 
                                        handle_outliers_vectorized(standardized_data, ['Income_Level', 'Coverage_Amount', 'Premium_Amount']).T)))
            log_detailed_dataframe_info(outlier_handled_data, "after handling outliers")

            imputed_data = impute_missing_values(outlier_handled_data)
            log_detailed_dataframe_info(imputed_data, "after imputing missing values")

            clustered_data, scaled_features = perform_kmeans_clustering(imputed_data)
            log_detailed_dataframe_info(clustered_data, "after performing K-means clustering")

            pca_data = perform_pca(clustered_data, scaled_features)
            log_detailed_dataframe_info(pca_data, "after performing PCA")

            risk_scored_data = calculate_risk_score(pca_data)
            log_detailed_dataframe_info(risk_scored_data, "after calculating risk score")

            encoded_data = encode_categorical_columns(risk_scored_data)
            log_detailed_dataframe_info(encoded_data, "after encoding categorical columns")

            transformed_data = encoded_data.astype({col: 'float32' for col in encoded_data.select_dtypes(include=['float64']).columns})
            log_detailed_dataframe_info(transformed_data, "after final transformation")

        except Exception as e:
            logger.error(f"Error in transform_data: {str(e)}")
            logger.error(f"Columns in raw_data: {raw_data.columns.tolist()}")
            raise

        transformed_validation = validate_data(transformed_data)
        
        if not validate_date_format(transformed_data):
            logger.error("Date format validation failed after transformation. Check 'Purchase History' column for inconsistencies.")

        logger.info(f"Number of rows in transformed data: {len(transformed_data)}")
        logger.info("Transformed Data Sample:\n%s", transformed_data.head())

        log_overall_performance(start_time, start_memory)

        return transformed_data, raw_validation, transformed_validation
    except Exception as e:
        logger.error(f"ETL process failed: {e}")
        raise

if __name__ == "__main__":
    try:
        config = load_config('config.yml')
        etl_process(config)
    except Exception as e:
        logger.error(f"Main process failed: {e}")