import pandas
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import yaml
import logging
import os
from etl_pipeline import etl_process

logging.basicConfig(
    filename='snowflake_connection.log',
    level=logging.DEBUG
)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data_to_snowflake(df, config):
    conn = snowflake.connector.connect(
        account=config['snowflake']['account'],
        user=config['snowflake']['user'],
        password=config['snowflake']['password'],
        warehouse=config['snowflake']['warehouse'],
        database=config['snowflake']['database'],
        schema=config['snowflake']['schema'],
        login_timeout=60,
        network_timeout=60
    )

    try:
        success, nchunks, nrows, _ = write_pandas(
            conn,
            df,
            'customer_segmentation',
            quote_identifiers=False
        )
        print(f"Data loaded successfully. Rows loaded: {nrows}")
    except Exception as e:
        print(f"Error loading data to Snowflake: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    config = load_config('config.yml')
    transformed_data, _, _ = etl_process(config)
    load_data_to_snowflake(transformed_data, config)
