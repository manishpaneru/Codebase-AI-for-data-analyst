"""
AI Ethics Analysis Tool
=====================

This script provides tools for analyzing AI systems for ethical considerations
and potential biases. It helps identify and mitigate ethical risks in AI applications.

Author: Sujal Dhungana , Manish Paneru
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.metrics import ClassificationMetric
from openai import OpenAI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EthicalAnalyzer:
    """
    A class that performs ethical analysis of datasets and models.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str, protected_attributes: List[str]):
        """
        Initialize the Ethical Analyzer.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of target variable column
            protected_attributes (List[str]): List of protected attribute columns
        """
        self.df = df.copy()
        self.target_column = target_column
        self.protected_attributes = protected_attributes
        self.metrics = {}
        self.recommendations = {}
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Create output directory
        self.output_path = Path('ethical_output')
        self.output_path.mkdir(exist_ok=True)
    
    def prepare_dataset(self) -> BinaryLabelDataset:
        """
        Prepare dataset for bias analysis.
        
        Returns:
            BinaryLabelDataset: Prepared dataset
        """
        # Convert to binary label dataset format
        df = self.df.copy()
        
        # Identify privileged and unprivileged values
        privileged_groups = []
        unprivileged_groups = []
        
        for attr in self.protected_attributes:
            # Assume binary protected attributes for simplicity
            values = sorted(df[attr].unique())
            if len(values) == 2:
                privileged_groups.append({attr: values[1]})
                unprivileged_groups.append({attr: values[0]})
        
        # Create dataset
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[self.target_column],
            protected_attribute_names=self.protected_attributes,
            privileged_protected_attributes=[1] * len(self.protected_attributes)
        )
        
        return dataset
    
    def calculate_bias_metrics(self, dataset: BinaryLabelDataset) -> Dict[str, float]:
        """
        Calculate various bias metrics.
        
        Args:
            dataset (BinaryLabelDataset): Prepared dataset
            
        Returns:
            Dict[str, float]: Calculated metrics
        """
        metrics = {}
        
        # Calculate disparate impact
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=[{attr: 0} for attr in self.protected_attributes],
            privileged_groups=[{attr: 1} for attr in self.protected_attributes]
        )
        
        metrics['disparate_impact'] = metric.disparate_impact()
        metrics['statistical_parity_difference'] = metric.statistical_parity_difference()
        metrics['equal_opportunity_difference'] = metric.equal_opportunity_difference()
        
        self.metrics = metrics
        return metrics
    
    def analyze_bias(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze bias metrics using AI.
        
        Args:
            metrics (Dict[str, float]): Calculated bias metrics
            
        Returns:
            Dict[str, Any]: Bias analysis results
        """
        analysis_context = {
            'metrics': metrics,
            'dataset_info': {
                'shape': self.df.shape,
                'protected_attributes': self.protected_attributes,
                'target_column': self.target_column
            }
        }
        
        prompt = f"""
        Analyze these bias metrics and provide insights:
        {json.dumps(analysis_context, indent=2)}
        
        Please provide:
        1. Assessment of bias presence
        2. Severity of identified biases
        3. Potential sources of bias
        4. Impact on different groups
        5. Ethical considerations
        
        Focus on actionable insights and ethical implications.
        Respond in JSON format with structured analysis.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an ethical AI expert. Provide clear, actionable insights about bias."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logging.error(f"Error analyzing bias: {str(e)}")
            return {"error": "Bias analysis failed"}
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations for bias mitigation.
        
        Args:
            analysis (Dict[str, Any]): Bias analysis results
            
        Returns:
            Dict[str, Any]: Recommendations
        """
        prompt = f"""
        Generate recommendations based on this bias analysis:
        {json.dumps(analysis, indent=2)}
        
        Please provide:
        1. Specific mitigation strategies
        2. Data collection improvements
        3. Model adjustments
        4. Process changes
        5. Monitoring suggestions
        
        Make recommendations specific and actionable.
        Respond in JSON format with structured recommendations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an ethical AI expert. Provide practical recommendations for bias mitigation."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            self.recommendations = recommendations
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return {"error": "Recommendation generation failed"}
    
    def mitigate_bias(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        """
        Apply bias mitigation techniques.
        
        Args:
            dataset (BinaryLabelDataset): Original dataset
            
        Returns:
            BinaryLabelDataset: Mitigated dataset
        """
        try:
            # Apply reweighing
            RW = Reweighing(
                unprivileged_groups=[{attr: 0} for attr in self.protected_attributes],
                privileged_groups=[{attr: 1} for attr in self.protected_attributes]
            )
            dataset_transformed = RW.fit_transform(dataset)
            
            return dataset_transformed
            
        except Exception as e:
            logging.error(f"Error in bias mitigation: {str(e)}")
            return dataset
    
    def evaluate_fairness(self, original_dataset: BinaryLabelDataset,
                         mitigated_dataset: BinaryLabelDataset) -> Dict[str, Any]:
        """
        Evaluate fairness metrics before and after mitigation.
        
        Args:
            original_dataset (BinaryLabelDataset): Original dataset
            mitigated_dataset (BinaryLabelDataset): Mitigated dataset
            
        Returns:
            Dict[str, Any]: Fairness evaluation results
        """
        results = {
            'original': self.calculate_bias_metrics(original_dataset),
            'mitigated': self.calculate_bias_metrics(mitigated_dataset)
        }
        
        # Calculate improvement
        results['improvement'] = {
            metric: results['mitigated'][metric] - results['original'][metric]
            for metric in results['original'].keys()
        }
        
        return results
    
    def create_visualizations(self) -> None:
        """
        Create interactive visualizations of bias analysis results.
        """
        st.title("Ethical Data Analysis Results")
        
        # 1. Bias Metrics Overview
        st.header("Bias Metrics")
        metrics_df = pd.DataFrame([self.metrics])
        
        fig = go.Figure()
        for metric, value in self.metrics.items():
            fig.add_trace(go.Bar(
                name=metric,
                x=[metric],
                y=[value]
            ))
        
        fig.update_layout(title="Bias Metrics Overview")
        st.plotly_chart(fig)
        
        # 2. Protected Attribute Distributions
        st.header("Protected Attribute Distributions")
        for attr in self.protected_attributes:
            fig = px.histogram(
                self.df,
                x=attr,
                color=self.target_column,
                title=f"Distribution of {attr} by {self.target_column}"
            )
            st.plotly_chart(fig)
        
        # 3. Fairness Metrics Comparison
        if hasattr(self, 'fairness_results'):
            st.header("Fairness Metrics Comparison")
            
            comparison_df = pd.DataFrame({
                'Original': self.fairness_results['original'],
                'Mitigated': self.fairness_results['mitigated']
            }).reset_index()
            
            fig = px.bar(
                comparison_df,
                x='index',
                y=['Original', 'Mitigated'],
                barmode='group',
                title="Fairness Metrics Before and After Mitigation"
            )
            st.plotly_chart(fig)
        
        # 4. Recommendations
        st.header("Recommendations")
        if self.recommendations:
            for category, items in self.recommendations.items():
                st.subheader(category.replace('_', ' ').title())
                if isinstance(items, list):
                    for item in items:
                        st.write(f"- {item}")
                elif isinstance(items, dict):
                    for key, value in items.items():
                        st.write(f"- {key}: {value}")
                else:
                    st.write(items)
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive ethical analysis report.
        
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_info': {
                'shape': self.df.shape,
                'protected_attributes': self.protected_attributes,
                'target_column': self.target_column
            },
            'bias_metrics': self.metrics,
            'recommendations': self.recommendations,
            'fairness_results': getattr(self, 'fairness_results', None)
        }
        
        prompt = f"""
        Generate an ethical analysis report based on these results:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Methodology
        3. Bias Analysis Results
        4. Fairness Evaluation
        5. Recommendations
        6. Ethical Considerations
        7. Next Steps
        
        Format the report in Markdown, including sections for visualizations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an ethical AI expert. Create comprehensive, actionable reports."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'ethical_analysis_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def run_analysis(self) -> Tuple[str, Path]:
        """
        Run the complete ethical analysis pipeline.
        
        Returns:
            Tuple[str, Path]: Analysis report and path to output directory
        """
        try:
            # 1. Prepare dataset
            dataset = self.prepare_dataset()
            
            # 2. Calculate bias metrics
            metrics = self.calculate_bias_metrics(dataset)
            
            # 3. Analyze bias
            analysis = self.analyze_bias(metrics)
            
            # 4. Generate recommendations
            self.generate_recommendations(analysis)
            
            # 5. Mitigate bias
            mitigated_dataset = self.mitigate_bias(dataset)
            
            # 6. Evaluate fairness
            self.fairness_results = self.evaluate_fairness(dataset, mitigated_dataset)
            
            # 7. Create visualizations
            self.create_visualizations()
            
            # 8. Generate report
            report = self.generate_report()
            
            return report, self.output_path
            
        except Exception as e:
            logging.error(f"Error in analysis pipeline: {str(e)}")
            return "Error in analysis", self.output_path

def main():
    """
    Main function to run the Ethical Analyzer.
    """
    try:
        st.title("AI-Powered Ethical Data Analysis")
        st.write("""
        This tool helps you identify and mitigate bias in your datasets,
        ensuring fair and ethical data analysis practices.
        """)
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Select columns
            target_column = st.selectbox(
                "Select target column",
                df.columns
            )
            
            protected_attributes = st.multiselect(
                "Select protected attributes",
                [col for col in df.columns if col != target_column]
            )
            
            if target_column and protected_attributes:
                # Initialize analyzer
                analyzer = EthicalAnalyzer(df, target_column, protected_attributes)
                
                if st.button("Analyze Ethics"):
                    with st.spinner("Performing ethical analysis..."):
                        # Run analysis
                        report, output_path = analyzer.run_analysis()
                        
                        # Display report
                        st.markdown(report)
                        
                        # Download button
                        st.download_button(
                            "Download Full Report",
                            report,
                            "ethical_analysis_report.md",
                            "text/markdown"
                        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 