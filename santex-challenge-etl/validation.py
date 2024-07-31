import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_date_format(df):
    if 'Purchase_History' not in df.columns:
        logger.error("'Purchase_History' column not found in the dataframe")
        return False
    
    if pd.api.types.is_datetime64_any_dtype(df['Purchase_History']):
        return True
    
    try:
        pd.to_datetime(df['Purchase_History'], format='%Y-%m-%d')
        return True
    except ValueError:
        invalid_dates = df[pd.to_datetime(df['Purchase_History'], format='%Y-%m-%d', errors='coerce').isnull()]
        if not invalid_dates.empty:
            logger.warning(f"Found {len(invalid_dates)} rows with invalid date format in 'Purchase_History'. Examples: {invalid_dates['Purchase_History'].head().tolist()}")
        return False

def validate_data(df, log_results=True):
    results = {
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_ranges': {col: {'min': df[col].min(), 'max': df[col].max()} 
                           for col in df.select_dtypes(include=['int32', 'float32', 'int64', 'float64']).columns},
        'unique_values': {col: df[col].unique().tolist() 
                          for col in df.select_dtypes(include=['category', 'object']).columns}
    }
    
    if log_results:
        logger.info("=== Data Validation Results ===")
        logger.info("1. Missing Values:")
        for col, count in results['missing_values'].items():
            logger.info(f"   {col}: {count}")
        
        logger.info("2. Data Types:")
        for col, dtype in results['data_types'].items():
            logger.info(f"   {col}: {dtype}")
        
        logger.info("3. Numeric Ranges:")
        for col, range_info in results['numeric_ranges'].items():
            logger.info(f"   {col}:")
            logger.info(f"      Min: {range_info['min']}")
            logger.info(f"      Max: {range_info['max']}")
        
        logger.info("4. Unique Values in Categorical Columns:")
        for col, unique_vals in results['unique_values'].items():
            logger.info(f"   {col}:")
            logger.info(f"      {', '.join(map(str, unique_vals[:5]))}")
            if len(unique_vals) > 5:
                logger.info(f"      ... and {len(unique_vals) - 5} more")
    
    return results

def log_dataframe_info(df: pd.DataFrame, stage: str):
    logger.info(f"DataFrame info at {stage}:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Sample data:\n{df.head().to_string()}")

def log_detailed_dataframe_info(df: pd.DataFrame, stage: str, num_rows: int = 10):
    logger.info(f"Detailed DataFrame info at {stage}:")
    
    # Set display options to show all columns and rows
    with pd.option_context('display.max_columns', None, 
                           'display.width', None, 
                           'display.max_rows', num_rows):
        
        logger.info(f"Sample data (first {num_rows} rows):\n{df.head(num_rows).to_string()}")
    
    logger.info("\nAll columns and their types:")
    for col in df.columns:
        logger.info(f"- {col} (Type: {df[col].dtype})")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    logger.info("\nBasic statistics for numeric columns:")
    logger.info(df[numeric_columns].describe().to_string())

    categorical_columns = df.select_dtypes(include=['category', 'object']).columns
    for col in categorical_columns:
        logger.info(f"\nTop 5 value counts for {col}:")
        logger.info(df[col].value_counts().head().to_string())

    logger.info(f"\nTotal number of rows: {len(df)}")
    logger.info(f"Total number of columns: {len(df.columns)}")