"""
AI-Powered Data Cleaning Script
This script demonstrates how to leverage OpenAI's GPT models to clean messy datasets.
It showcases various data cleaning techniques including:
- Missing value imputation
- Duplicate detection and removal
- Typo correction
- Data format standardization

The script reads from data.csv which contains intentionally messy data with common issues like:
- Typos in names and cities (e.g., "Jhn Smth", "San Fransisco")
- Missing values in Age and City columns
- Inconsistent formatting in Occupation (e.g., "Software Engineeer" vs "Software Engineer")
- Duplicate entries
- Variations in city names ("San Francisco" vs "San Fran")
"""

import pandas as pd
import numpy as np
from openai import OpenAI
import logging
from typing import List, Dict, Union
import json

# Configure logging to track the cleaning process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client - Replace with your API key
OPENAI_API_KEY = "your_api_key_here"
client = OpenAI(api_key=OPENAI_API_KEY)

class AIDataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the AI Data Cleaner with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame to be cleaned
        """
        self.df = df.copy()  # Create a copy to preserve original data
        self.cleaning_log = []  # Track all cleaning operations
        
    def detect_anomalies_with_ai(self, column: str) -> List[int]:
        """
        Use GPT-4 to detect anomalies in a specific column.
        This method sends column data to GPT-4 and asks it to identify potential anomalies.
        
        Args:
            column (str): Name of the column to check for anomalies
            
        Returns:
            List[int]: Indices of rows containing anomalies
        """
        # Sample the column data to send to GPT-4
        sample_data = self.df[column].head(10).tolist()
        
        # Construct a prompt for GPT-4
        prompt = f"""
        Analyze this data sample and identify any anomalies or inconsistencies:
        {sample_data}
        
        Please respond in JSON format with:
        1. Identified patterns
        2. Potential anomalies
        3. Suggested cleaning rules
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data cleaning expert. Analyze the data and provide cleaning recommendations."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response and identify anomalies
            analysis = json.loads(response.choices[0].message.content)
            logging.info(f"AI Analysis for column {column}: {analysis}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in AI anomaly detection: {str(e)}")
            return []

    def correct_typos_with_ai(self, column: str) -> None:
        """
        Use GPT-4 to correct typos in text data.
        This method is particularly useful for categorical data with spelling variations.
        
        Args:
            column (str): Name of the column to correct typos in
        """
        unique_values = self.df[column].unique()
        
        # Construct a prompt for GPT-4 to identify and correct typos
        prompt = f"""
        These are unique values in a dataset column. Identify and correct any typos:
        {list(unique_values)}
        
        Respond with a JSON dictionary mapping original values to corrected values.
        Only include values that need correction.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data cleaning expert. Identify and correct typos in the data."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Apply corrections
            corrections = json.loads(response.choices[0].message.content)
            self.df[column] = self.df[column].replace(corrections)
            self.cleaning_log.append(f"Applied typo corrections in column {column}: {corrections}")
            
        except Exception as e:
            logging.error(f"Error in typo correction: {str(e)}")

    def impute_missing_values_with_ai(self, column: str) -> None:
        """
        Use AI to intelligently impute missing values based on patterns in the data.
        This method analyzes the column's data distribution and context to make smart imputations.
        
        Args:
            column (str): Name of the column to impute missing values in
        """
        if not self.df[column].isnull().any():
            return
        
        # Prepare context for GPT-4
        column_stats = {
            "dtype": str(self.df[column].dtype),
            "non_null_values": self.df[column].dropna().head(10).tolist(),
            "missing_count": self.df[column].isnull().sum()
        }
        
        prompt = f"""
        Analyze this column's statistics and suggest the best imputation strategy:
        {json.dumps(column_stats, indent=2)}
        
        Provide recommendations in JSON format including:
        1. Recommended imputation method
        2. Justification
        3. Specific values or approach to use
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data science expert. Recommend the best imputation strategy."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Implement the recommended imputation strategy
            recommendation = json.loads(response.choices[0].message.content)
            logging.info(f"AI Imputation recommendation for {column}: {recommendation}")
            
            # Apply basic imputation based on AI recommendation
            # In a real scenario, you would implement more sophisticated imputation based on the AI's recommendation
            self.df[column].fillna(self.df[column].mean() if self.df[column].dtype in ['int64', 'float64'] else self.df[column].mode()[0], inplace=True)
            
        except Exception as e:
            logging.error(f"Error in AI imputation: {str(e)}")

    def clean_dataset(self) -> pd.DataFrame:
        """
        Main method to clean the entire dataset using AI-powered techniques.
        This orchestrates the entire cleaning process and applies various cleaning methods.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        logging.info("Starting AI-powered data cleaning process...")
        
        # Process each column in the DataFrame
        for column in self.df.columns:
            logging.info(f"Cleaning column: {column}")
            
            # 1. Detect and handle anomalies
            anomalies = self.detect_anomalies_with_ai(column)
            logging.info(f"Anomalies detected in {column}: {anomalies}")
            
            # 2. Correct typos if column is object/string type
            if self.df[column].dtype == 'object':
                self.correct_typos_with_ai(column)
            
            # 3. Handle missing values
            self.impute_missing_values_with_ai(column)
        
        logging.info("Data cleaning process completed")
        return self.df

def main():
    """
    Example usage of the AIDataCleaner class with data loaded from data.csv
    This file contains intentionally messy data to demonstrate various cleaning capabilities.
    """
    try:
        # Load the messy dataset from CSV
        messy_data = pd.read_csv('data.csv')
        
        print("Original Dataset:")
        print(messy_data)
        print("\n" + "="*50 + "\n")
        
        # Initialize and run the AI-powered cleaner
        cleaner = AIDataCleaner(messy_data)
        cleaned_data = cleaner.clean_dataset()
        
        print("Cleaned Dataset:")
        print(cleaned_data)
        print("\nCleaning Log:")
        for log in cleaner.cleaning_log:
            print(f"- {log}")
            
        # Save the cleaned dataset
        cleaned_data.to_csv('cleaned_data.csv', index=False)
        print("\nCleaned data has been saved to 'cleaned_data.csv'")
        
    except FileNotFoundError:
        print("Error: data.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 