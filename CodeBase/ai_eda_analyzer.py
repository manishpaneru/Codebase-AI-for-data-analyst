"""
AI-Powered Exploratory Data Analysis (EDA) Tool
This script leverages OpenAI's GPT models to perform intelligent exploratory data analysis.
It automatically generates:
- Statistical summaries
- Data quality reports
- Relevant visualizations
- Natural language insights
- Correlation analysis
- Distribution analysis
- Outlier detection

The script uses AI to determine the most appropriate visualizations and analysis methods
based on the data types and patterns it discovers in your dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import logging
from typing import List, Dict, Union, Tuple
import json
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client - Replace with your API key
OPENAI_API_KEY = "your_api_key_here"
client = OpenAI(api_key=OPENAI_API_KEY)

class AIDataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the AI-powered Data Analyzer.
        
        Args:
            df (pd.DataFrame): The input DataFrame to analyze
        """
        self.df = df.copy()
        self.insights = []  # Store AI-generated insights
        self.analysis_path = Path('analysis_output')  # Directory for saving visualizations
        self.analysis_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def generate_basic_stats(self) -> Dict:
        """
        Generate basic statistical information about the dataset.
        Uses AI to interpret the statistics and provide human-readable insights.
        
        Returns:
            Dict: Dictionary containing statistical information and AI interpretations
        """
        # Calculate basic statistics
        stats = {
            'basic_stats': self.df.describe().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns}
        }
        
        # Get AI interpretation of the statistics
        prompt = f"""
        Analyze these dataset statistics and provide key insights:
        {json.dumps(stats, indent=2)}
        
        Please provide insights about:
        1. Data quality issues
        2. Notable patterns in the statistics
        3. Potential areas for further investigation
        4. Recommendations for data preprocessing
        
        Respond in JSON format with structured insights.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Interpret these statistics and provide valuable insights."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            stats['ai_interpretation'] = json.loads(response.choices[0].message.content)
            self.insights.append(('Statistical Analysis', stats['ai_interpretation']))
            
        except Exception as e:
            logging.error(f"Error in AI statistical interpretation: {str(e)}")
            stats['ai_interpretation'] = {"error": "AI interpretation failed"}
            
        return stats
    
    def analyze_distributions(self) -> None:
        """
        Analyze and visualize the distribution of each numerical column.
        Uses AI to determine the best visualization type and generate insights.
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            
            # Create distribution plot
            sns.histplot(data=self.df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(self.analysis_path / f'distribution_{col}.png')
            plt.close()
            
            # Get AI insights about the distribution
            stats = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'skew': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis()),
                'q1': float(self.df[col].quantile(0.25)),
                'q3': float(self.df[col].quantile(0.75))
            }
            
            prompt = f"""
            Analyze this distribution for column '{col}':
            {json.dumps(stats, indent=2)}
            
            Provide insights about:
            1. Shape of the distribution
            2. Presence of outliers
            3. Skewness and what it means for this data
            4. Recommendations for handling this distribution
            
            Respond in JSON format.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a statistical analysis expert. Interpret this distribution and provide insights."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                insights = json.loads(response.choices[0].message.content)
                self.insights.append((f'Distribution Analysis - {col}', insights))
                
            except Exception as e:
                logging.error(f"Error in distribution analysis for {col}: {str(e)}")
    
    def analyze_correlations(self) -> None:
        """
        Analyze correlations between numerical variables and generate insights.
        Creates correlation heatmap and uses AI to interpret relationships.
        """
        numerical_df = self.df.select_dtypes(include=[np.number])
        if len(numerical_df.columns) < 2:
            return
        
        # Create correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.analysis_path / 'correlation_heatmap.png')
        plt.close()
        
        # Get AI insights about correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    strong_correlations.append({
                        'variables': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        
        prompt = f"""
        Analyze these correlations:
        {json.dumps(strong_correlations, indent=2)}
        
        Provide insights about:
        1. Strongest relationships found
        2. Potential causation vs correlation
        3. Recommendations for feature engineering
        4. Potential business implications
        
        Respond in JSON format.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a correlation analysis expert. Interpret these relationships and provide insights."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            insights = json.loads(response.choices[0].message.content)
            self.insights.append(('Correlation Analysis', insights))
            
        except Exception as e:
            logging.error(f"Error in correlation analysis: {str(e)}")
    
    def analyze_categorical_relationships(self) -> None:
        """
        Analyze relationships between categorical variables and their impact on numerical variables.
        Creates appropriate visualizations and generates AI insights.
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                plt.figure(figsize=(12, 6))
                
                # Create box plot
                sns.boxplot(data=self.df, x=cat_col, y=num_col)
                plt.xticks(rotation=45)
                plt.title(f'{num_col} by {cat_col}')
                plt.tight_layout()
                plt.savefig(self.analysis_path / f'boxplot_{cat_col}_{num_col}.png')
                plt.close()
                
                # Get AI insights about the relationship
                stats = {
                    'category_counts': self.df[cat_col].value_counts().to_dict(),
                    'means_by_category': self.df.groupby(cat_col)[num_col].mean().to_dict(),
                    'medians_by_category': self.df.groupby(cat_col)[num_col].median().to_dict()
                }
                
                prompt = f"""
                Analyze the relationship between categorical variable '{cat_col}' 
                and numerical variable '{num_col}':
                {json.dumps(stats, indent=2)}
                
                Provide insights about:
                1. Differences between categories
                2. Potential outliers or unusual patterns
                3. Business implications of these differences
                4. Recommendations for further analysis
                
                Respond in JSON format.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a data analysis expert. Interpret these categorical relationships and provide insights."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    insights = json.loads(response.choices[0].message.content)
                    self.insights.append((f'Categorical Analysis - {cat_col} vs {num_col}', insights))
                    
                except Exception as e:
                    logging.error(f"Error in categorical analysis for {cat_col} vs {num_col}: {str(e)}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report using AI.
        Combines all insights and creates a narrative summary.
        
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_shape': self.df.shape,
            'columns': list(self.df.columns),
            'insights': self.insights
        }
        
        prompt = f"""
        Generate a comprehensive data analysis report based on these insights:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis by Category
        4. Recommendations
        5. Next Steps
        
        Format the report in Markdown.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data science report writer. Create a comprehensive analysis report."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report to file
            with open(self.analysis_path / 'analysis_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error in report generation: {str(e)}")
            return "Error generating report"
    
    def run_complete_analysis(self) -> Tuple[str, Path]:
        """
        Run the complete EDA process and generate all insights and visualizations.
        
        Returns:
            Tuple[str, Path]: Analysis report and path to output directory
        """
        logging.info("Starting AI-powered EDA process...")
        
        # Run all analyses
        self.generate_basic_stats()
        self.analyze_distributions()
        self.analyze_correlations()
        self.analyze_categorical_relationships()
        
        # Generate final report
        report = self.generate_report()
        
        logging.info(f"Analysis complete. Output saved to {self.analysis_path}")
        return report, self.analysis_path

def main():
    """
    Example usage of the AIDataAnalyzer class
    """
    try:
        # Load the dataset
        df = pd.read_csv('data.csv')
        
        print("Starting AI-Powered EDA...")
        analyzer = AIDataAnalyzer(df)
        report, output_path = analyzer.run_complete_analysis()
        
        print(f"\nAnalysis complete! Check the following locations for results:")
        print(f"- Report: {output_path / 'analysis_report.md'}")
        print(f"- Visualizations: {output_path}")
        print("\nReport Preview:")
        print("="*50)
        print(report[:500] + "...\n")  # Show first 500 characters of the report
        
    except FileNotFoundError:
        print("Error: data.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 