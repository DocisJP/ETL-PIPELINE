import unittest
import pandas as pd
import numpy as np
from validation import validate_date_format, validate_data
from etl_pipeline import (transform_data, extract_data_from_s3, preprocess_data, encode_categorical_columns, 
                          add_time_based_features, standardize_categorical_formats,
                          handle_outliers_vectorized, impute_missing_values,
                          perform_kmeans_clustering, perform_pca, calculate_risk_score, load_config)
from data_versioning import check_data_format_version, preprocess_data, add_time_based_features, reset_version_cache


class TestETLFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config('config.yml')
        cls.full_df = extract_data_from_s3(cls.config['s3']['bucket_name'], cls.config['s3']['file_key'])
        cls.total_rows = len(cls.full_df)
        print(f"\nTotal number of rows in the full dataset: {cls.total_rows}")
        
        # Use half of the data for testing
        cls.df = cls.full_df.sample(frac=0.5, random_state=42)
        cls.sample_rows = len(cls.df)
        print(f"Number of rows in the sample: {cls.sample_rows}")
        
        cls.rows_after_dropna = len(cls.df.dropna())
        print(f"Number of rows after dropna: {cls.rows_after_dropna}")
        print(f"Percentage of rows retained in sample: {(cls.rows_after_dropna / cls.sample_rows) * 100:.2f}%\n")
        
        cls.transformed_data = transform_data(cls.df)
        cls.raw_validation = validate_data(cls.df, log_results=False)
        cls.transformed_validation = validate_data(cls.transformed_data, log_results=False)
    
    def setUp(self):
        reset_version_cache()

    
    def test_data_validation(self):
        # Print actual types for debugging
        actual_types = self.transformed_data.dtypes.astype(str).to_dict()
        print("Actual types:", actual_types)

        # Test for missing values
        self.assertTrue(self.transformed_data.notnull().all().all(), "There are missing values in the transformed data")

        # Test for correct data types
        expected_types = {
            'Customer_ID': 'int64',
            'Age': 'int32',
            'Gender': 'category',
            'Marital_Status': 'category',
            'Education_Level': 'category',
            'Geographic_Information': 'category',
            'Occupation': 'category',
            'Income_Level': 'float32',
            'Behavioral_Data': 'category',
            'Purchase_History': 'object',
            'Interactions_with_Customer_Service': 'category',
            'Insurance_Products_Owned': 'category',
            'Coverage_Amount': 'float32',
            'Premium_Amount': 'float32',
            'Policy_Type': 'category',
            'Customer_Preferences': 'category',
            'Preferred_Communication_Channel': 'category',
            'Preferred_Contact_Time': 'category',
            'Preferred_Language': 'category',
            'Segmentation_Group': 'category',
            'Age_Group': 'category',
            'Purchase_Day_of_Week': 'int8',
            'Purchase_Month': 'int8',
            'Purchase_Season': 'category',
            'Customer_Tenure': 'int32',
            'Cluster': 'int64',
            'PCA1': 'float32',
            'PCA2': 'float32',
            'Risk_Score': 'float32'
        }
        self.assertDictEqual(expected_types, actual_types, "Data types do not match expected types")

        # Test for numeric ranges
        numeric_ranges = {
            'Age': {'min': 18, 'max': 100},
            'Income_Level': {'min': 0, 'max': 1},
            'Coverage_Amount': {'min': 0},
            'Premium_Amount': {'min': 0},
            'Customer_Tenure': {'min': 0},
            'Risk_Score': {'min': 0, 'max': 1}
        }
        for col, range_vals in numeric_ranges.items():
            if 'min' in range_vals:
                self.assertTrue((self.transformed_data[col] >= range_vals['min']).all(), 
                                f"{col} contains values below the minimum expected value")
            if 'max' in range_vals:
                self.assertTrue((self.transformed_data[col] <= range_vals['max']).all(), 
                                f"{col} contains values above the maximum expected value")

        # Test for categorical values
        expected_categories = {
            'Gender': {'Male', 'Female'},
            'Marital_Status': {'Single', 'Married', 'Divorced', 'Widowed', 'Separated'},
            'Education_Level': {'High School Diploma', "Associate's Degree", "Bachelor's Degree", "Master's Degree", 'Doctorate'},
            'Occupation': {'Manager', 'Engineer', 'Teacher', 'Salesperson', 'Entrepreneur', 'Doctor', 'Lawyer', 'Artist', 'Nurse'},
            'Behavioral_Data': {'Policy1', 'Policy2', 'Policy3', 'Policy4', 'Policy5'},
            'Interactions_with_Customer_Service': {'Phone', 'Email', 'Chat', 'In-Person', 'Mobile App'},
            'Insurance_Products_Owned': {'Policy1', 'Policy2', 'Policy3', 'Policy4', 'Policy5'},
            'Policy_Type': {'Individual', 'Family', 'Group', 'Business'},
            'Customer_Preferences': {'Email', 'Phone', 'Mail', 'Text', 'In-Person Meeting'},
            'Preferred_Communication_Channel': {'Email', 'Phone', 'Mail', 'Text', 'In-Person Meeting'},
            'Preferred_Contact_Time': {'Morning', 'Afternoon', 'Evening', 'Anytime', 'Weekends'},
            'Preferred_Language': {'English', 'Spanish', 'French', 'German', 'Mandarin'},
            'Segmentation_Group': {'Segment1', 'Segment2', 'Segment3', 'Segment4', 'Segment5'},
            'Age_Group': {'0-18', '19-30', '31-50', '51-65', '65+'},
            'Purchase_Season': {'Spring', 'Summer', 'Fall', 'Winter'}
        }
        for col, expected_vals in expected_categories.items():
            actual_vals = set(self.transformed_data[col].unique())
            unexpected_vals = actual_vals - expected_vals
            self.assertFalse(unexpected_vals, f"Unexpected values in {col}: {unexpected_vals}")

        # Test date format
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        self.assertTrue(self.transformed_data['Purchase_History'].str.match(date_pattern).all(), 
                        "Purchase_History contains invalid date formats")

    def test_data_format_version_detection(self):
        df_v1 = self.df.copy()
        self.assertEqual(check_data_format_version(df_v1), 1)

        reset_version_cache()
        df_v2 = self.df.copy()
        df_v2['New_Column'] = 'some_value'
        self.assertEqual(check_data_format_version(df_v2), 2)

    def test_preprocess_data(self):
        # Test for version 1 (standard processing)
        transformed_df_v1 = self.transformed_data
        self.assertIn('Age', transformed_df_v1.columns)
        self.assertEqual(transformed_df_v1['Age'].dtype, 'int32')
        self.assertIn('Age_Group', transformed_df_v1.columns)
        self.assertEqual(transformed_df_v1['Age_Group'].dtype, 'category')
        self.assertNotIn('New_Column', transformed_df_v1.columns)

        # Test for version 2 (with New_Column)
        raw_df_v2 = self.full_df.copy()
        raw_df_v2['New Column'] = 'some_value'
        transformed_df_v2 = transform_data(raw_df_v2)
        
        self.assertIn('Age', transformed_df_v2.columns)
        self.assertEqual(transformed_df_v2['Age'].dtype, 'int32')
        self.assertIn('Age_Group', transformed_df_v2.columns)
        self.assertEqual(transformed_df_v2['Age_Group'].dtype, 'category')
        self.assertIn('New_Column', transformed_df_v2.columns)
        
        # Check if New_Column is processed correctly
        if 'New_Column' in transformed_df_v2.columns:
            self.assertTrue(transformed_df_v2['New_Column'].notna().all(), "New_Column should not contain NaN values")
            self.assertTrue((transformed_df_v2['New_Column'] == 'some_value').all(), "New_Column should contain 'some_value'")

        # Test for differences between versions
        self.assertNotEqual(set(transformed_df_v1.columns), set(transformed_df_v2.columns), 
                            "Version 1 and Version 2 should have different columns")

    def test_add_time_based_features(self):
        time_df_v1 = add_time_based_features(self.df)
        self.assertIn('Purchase_Day_of_Week', time_df_v1.columns)
        self.assertIn('Purchase_Month', time_df_v1.columns)
        self.assertNotIn('Purchase_Year', time_df_v1.columns)

        reset_version_cache()
        df_v2 = self.df.copy()
        df_v2['New_Column'] = 'some_value'
        time_df_v2 = add_time_based_features(df_v2)
        self.assertIn('Purchase_Day_of_Week', time_df_v2.columns)
        self.assertIn('Purchase_Month', time_df_v2.columns)
        self.assertIn('Purchase_Year', time_df_v2.columns)

    def test_transformed_data_validation(self):
        transformed_df = transform_data(self.df)
        self.assertTrue(validate_date_format(transformed_df), "Date format validation failed for transformed data")

    def test_encode_categorical_columns(self):
        encoded_df = encode_categorical_columns(self.df)
        self.assertEqual(encoded_df['Gender'].dtype, 'category')

    def test_add_time_based_features(self):
        self.assertIn('Purchase_Day_of_Week', self.transformed_data.columns)
        self.assertIn('Purchase_Month', self.transformed_data.columns)
        self.assertIn('Purchase_Season', self.transformed_data.columns)
        self.assertIn('Customer_Tenure', self.transformed_data.columns)
        
        # Check if all dates are in 'YYYY-MM-DD' format
        self.assertTrue(self.transformed_data['Purchase_History'].str.match(r'^\d{4}-\d{2}-\d{2}$').all())

    def test_standardize_categorical_formats(self):
        std_df = standardize_categorical_formats(self.df)
        self.assertTrue(all(std_df['Gender'].str.istitle()))

    def test_handle_outliers_vectorized(self):
        numeric_columns = ['Income_Level', 'Coverage_Amount', 'Premium_Amount']
        for col in numeric_columns:
            self.assertIn(col, self.transformed_data.columns)
            self.assertFalse(self.transformed_data[col].isnull().any())

    def test_impute_missing_values(self):
        numeric_columns = ['Income_Level', 'Coverage_Amount', 'Premium_Amount']
        for col in numeric_columns:
            self.assertFalse(self.transformed_data[col].isnull().any())

    def test_perform_kmeans_clustering(self):
        self.assertIn('Cluster', self.transformed_data.columns)
        self.assertEqual(self.transformed_data['Cluster'].nunique(), 5)

    def test_perform_pca(self):
        self.assertIn('PCA1', self.transformed_data.columns)
        self.assertIn('PCA2', self.transformed_data.columns)

    def test_calculate_risk_score(self):
        self.assertIn('Risk_Score', self.transformed_data.columns)
        self.assertTrue((self.transformed_data['Risk_Score'] >= 0).all() and (self.transformed_data['Risk_Score'] <= 1).all())

    def test_end_to_end_transformation(self):
        # Test the entire transformation process for version 1
        transformed_df_v1 = transform_data(self.df)
        self.assertIsInstance(transformed_df_v1, pd.DataFrame)
        self.assertNotIn('Purchase_Year', transformed_df_v1.columns)

        # Reset cache before testing version 2
        reset_version_cache()
        
        # Test the entire transformation process for version 2
        df_v2 = self.df.copy()
        df_v2['New_Column'] = 'some_value'
        transformed_df_v2 = transform_data(df_v2)
        self.assertIsInstance(transformed_df_v2, pd.DataFrame)
        self.assertIn('Purchase_Year', transformed_df_v2.columns)

        expected_columns = ['Customer_ID', 'Age', 'Income_Level', 'Coverage_Amount', 'Premium_Amount',
                            'Purchase_History', 'Gender', 'Age_Group', 'Purchase_Day_of_Week',
                            'Purchase_Month', 'Purchase_Season', 'Customer_Tenure', 'Cluster',
                            'PCA1', 'PCA2', 'Risk_Score']
        for col in expected_columns:
            self.assertIn(col, transformed_df_v1.columns)
            self.assertIn(col, transformed_df_v2.columns)

if __name__ == '__main__':
    unittest.main()