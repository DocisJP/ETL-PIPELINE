-- Create a warehouse
CREATE WAREHOUSE IF NOT EXISTS customer_segmentation_wh
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE;

-- Create database
CREATE DATABASE IF NOT EXISTS customer_segmentation_db;

-- Use the created database
USE DATABASE customer_segmentation_db;

-- Create schema
CREATE SCHEMA IF NOT EXISTS etl_schema;

-- Use the created schema
USE SCHEMA etl_schema;

CREATE OR REPLACE TABLE customer_segmentation (
    Customer_ID INT,
    Age INT,
    Gender VARCHAR(10),
    Marital_Status VARCHAR(20),
    Education_Level VARCHAR(50),
    Geographic_Information VARCHAR(50),
    Occupation VARCHAR(50),
    Income_Level FLOAT,
    Behavioral_Data VARCHAR(20),
    Purchase_History DATE,
    Interactions_with_Customer_Service VARCHAR(20),
    Insurance_Products_Owned VARCHAR(20),
    Coverage_Amount FLOAT,
    Premium_Amount FLOAT,
    Policy_Type VARCHAR(20),
    Customer_Preferences VARCHAR(50),
    Preferred_Communication_Channel VARCHAR(50),
    Preferred_Contact_Time VARCHAR(20),
    Preferred_Language VARCHAR(20),
    Segmentation_Group VARCHAR(20),
    Age_Group VARCHAR(10),
    Purchase_Day_of_Week INT,
    Purchase_Month INT,
    Purchase_Season VARCHAR(10),
    Customer_Tenure INT,
    Cluster INT,
    PCA1 FLOAT,
    PCA2 FLOAT,
    Risk_Score FLOAT
);

ALTER TABLE customer_segmentation CLUSTER BY (segmentation_group, cluster);

ALTER TABLE customer_segmentation SUSPEND RECLUSTER;
ALTER TABLE customer_segmentation RESUME RECLUSTER;

