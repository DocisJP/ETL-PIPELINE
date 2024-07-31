USE DATABASE customer_segmentation_db;
USE SCHEMA etl_schema;

SELECT COUNT(*) AS total_records FROM customer_segmentation;

SELECT Cluster, COUNT(*) AS cluster_count
FROM customer_segmentation
GROUP BY Cluster
ORDER BY Cluster;

SELECT segmentation_group, cluster, AVG(risk_score) as calculated_avg_risk_score
FROM customer_segmentation
GROUP BY segmentation_group, cluster
ORDER BY segmentation_group, cluster;

SELECT 
    MIN(Purchase_History) AS earliest_purchase,
    MAX(Purchase_History) AS latest_purchase
FROM customer_segmentation;

SELECT Segmentation_Group, COUNT(*) AS group_count
FROM customer_segmentation
GROUP BY Segmentation_Group
ORDER BY Segmentation_Group;

SELECT 
    SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS null_customer_id,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS null_age,
    SUM(CASE WHEN Income_Level IS NULL THEN 1 ELSE 0 END) AS null_income_level,
    SUM(CASE WHEN Risk_Score IS NULL THEN 1 ELSE 0 END) AS null_risk_score
FROM customer_segmentation;

SELECT 
    MIN(Age) AS min_age, MAX(Age) AS max_age,
    MIN(Income_Level) AS min_income, MAX(Income_Level) AS max_income,
    MIN(Coverage_Amount) AS min_coverage, MAX(Coverage_Amount) AS max_coverage,
    MIN(Premium_Amount) AS min_premium, MAX(Premium_Amount) AS max_premium,
    MIN(Risk_Score) AS min_risk_score, MAX(Risk_Score) AS max_risk_score
FROM customer_segmentation;

SELECT Gender, COUNT(*) AS gender_count
FROM customer_segmentation
GROUP BY Gender
ORDER BY gender_count DESC;

SELECT Marital_Status, COUNT(*) AS marital_status_count
FROM customer_segmentation
GROUP BY Marital_Status
ORDER BY marital_status_count DESC;

SELECT Education_Level, COUNT(*) AS education_level_count
FROM customer_segmentation
GROUP BY Education_Level
ORDER BY education_level_count DESC;

SELECT 
    CORR(Risk_Score, Age) AS risk_age_correlation,
    CORR(Risk_Score, Income_Level) AS risk_income_correlation,
    CORR(Risk_Score, Coverage_Amount) AS risk_coverage_correlation,
    CORR(Risk_Score, Premium_Amount) AS risk_premium_correlation
FROM customer_segmentation;

SELECT 
    Purchase_Day_of_Week,
    COUNT(*) AS day_count
FROM customer_segmentation
GROUP BY Purchase_Day_of_Week
ORDER BY Purchase_Day_of_Week;

SELECT 
    Cluster,
    AVG(Risk_Score) AS avg_risk_score,
    MIN(Risk_Score) AS min_risk_score,
    MAX(Risk_Score) AS max_risk_score
FROM customer_segmentation
GROUP BY Cluster
ORDER BY Cluster;