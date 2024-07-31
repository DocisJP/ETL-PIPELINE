import pandas as pd
import numpy as np
from typing import Callable
import logging
from functools import wraps
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store the cached version
_cached_version = None

def check_data_format_version(df: pd.DataFrame) -> int:
    """
    Determine the version of the input data based on its structure.
    """
    global _cached_version
    if _cached_version is None:
        if 'New Column' in df.columns:
            _cached_version = 2
        else:
            _cached_version = 1
    logger.info(f"Detected data format version: {_cached_version}")
    return _cached_version

def reset_version_cache():
    """
    Reset the cached version. Call this when starting a new ETL process.
    """
    global _cached_version
    _cached_version = None
    logger.info("Version cache reset")

def version_handler(func: Callable) -> Callable:
    """
    Decorator to make functions version-aware.
    """
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        version = check_data_format_version(df)
        logger.info(f"Executing {func.__name__} with version {version}")
        return func(df, *args, **kwargs, version=version)
    return wrapper

def parse_date(date_string):
    date_formats = ['%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y']
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            pass
    
    # If standard formats fail, try a more flexible approach
    try:
        return pd.to_datetime(date_string)
    except ValueError:
        logger.warning(f"Unable to parse date: {date_string}")
        return pd.NaT

@version_handler
def preprocess_data(df: pd.DataFrame, version: int = 1) -> pd.DataFrame:
    logger.info("Starting data preprocessing")
    df = df.copy() 
    
    logger.info("Handling Purchase History dates")
    df['Purchase History'] = df['Purchase History'].apply(parse_date)
    
    unparseable_dates = df['Purchase History'].isnull().sum()
    logger.info(f"Number of unparseable dates: {unparseable_dates}")
    
    if unparseable_dates > 0:
        earliest_date = df['Purchase History'].min()
        df['Purchase History'] = df['Purchase History'].fillna(earliest_date)
        logger.info(f"Filled {unparseable_dates} unparseable dates with earliest date: {earliest_date}")
        
        # Create a flag for date quality
        df['Date_Quality'] = np.where(df['Purchase History'].isnull(), 'Imputed', 'Original')
    
    # Ensure all dates are in 'YYYY-MM-DD' format
    df['Purchase History'] = df['Purchase History'].dt.strftime('%Y-%m-%d')
    logger.info("All dates converted to YYYY-MM-DD format")

    logger.info("Processing numeric columns")
    float32_columns = ['Income Level', 'Coverage Amount', 'Premium Amount']
    df[float32_columns] = df[float32_columns].apply(pd.to_numeric, errors='coerce').astype('float32')

    logger.info("Normalizing Income Level")
    df['Income Level'] = (df['Income Level'] - df['Income Level'].min()) / (df['Income Level'].max() - df['Income Level'].min())

    logger.info("Processing Age column")
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('int32')
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 65, 100], labels=['0-18', '19-30', '31-50', '51-65', '65+'])

    logger.info("Handling categorical columns")
    categorical_columns = ['Gender', 'Marital Status', 'Education Level', 'Geographic Information', 
                           'Occupation', 'Behavioral Data', 'Interactions with Customer Service', 
                           'Insurance Products Owned', 'Policy Type', 'Customer Preferences', 
                           'Preferred Communication Channel', 'Preferred Contact Time', 
                           'Preferred Language', 'Segmentation Group']
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            logger.warning(f"Column {col} not found in the dataframe")

    if version == 2:
        logger.info("Applying version 2 specific processing")
        if 'New Column' in df.columns:
            df['New Column'] = df['New Column'].fillna('Unknown')

    logger.info("Ensuring column names are SQL-friendly")
    df.columns = df.columns.str.replace(' ', '_')

    logger.info("Data preprocessing completed")
    return df

@version_handler
def add_time_based_features(df: pd.DataFrame, version: int = 1) -> pd.DataFrame:
    logger.info("Adding time-based features")
    df['Purchase_History'] = pd.to_datetime(df['Purchase_History'], format='%Y-%m-%d', errors='coerce')
    
    invalid_dates = df['Purchase_History'].isnull().sum()
    if invalid_dates > 0:
        logger.warning(f"{invalid_dates} dates in 'Purchase_History' could not be parsed. Replacing with median date.")
    
    median_date = df['Purchase_History'].median()
    df['Purchase_History'] = df['Purchase_History'].fillna(median_date)
    
    df['Purchase_Day_of_Week'] = df['Purchase_History'].dt.dayofweek.astype('int8')
    df['Purchase_Month'] = df['Purchase_History'].dt.month.astype('int8')
    
    season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 
                  6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 
                  11: 'Fall', 12: 'Winter'}
    df['Purchase_Season'] = df['Purchase_Month'].map(season_map).astype('category')
    
    now = pd.Timestamp.now()
    df['Customer_Tenure'] = (now - df['Purchase_History']).dt.days.astype('int32')
    
    if version == 2:
        logger.info("Adding Purchase_Year column for version 2")
        df['Purchase_Year'] = df['Purchase_History'].dt.year.astype('int16')
    
    df['Purchase_History'] = df['Purchase_History'].dt.strftime('%Y-%m-%d')
    
    logger.info("Time-based features added successfully")
    return df
