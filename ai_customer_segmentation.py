"""
AI-Powered Customer Segmentation Tutorial
=====================================

This script demonstrates how to build an intelligent customer segmentation system
that uses clustering algorithms and NLP for customer profile generation.

Features:
1. Data preprocessing
2. Feature engineering
3. Clustering analysis
4. Profile generation
5. Segment visualization
6. Insights generation
7. Interactive dashboard

Author: [Your Name]
Date: [Current Date]
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from openai import OpenAI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomerSegmentation:
    """
    A class that performs AI-powered customer segmentation and profile generation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Customer Segmentation system.
        
        Args:
            df (pd.DataFrame): Customer dataset
        """
        self.df = df.copy()
        self.preprocessed_data = None
        self.clusters = None
        self.profiles = {}
        self.pca_components = None
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Create output directory
        self.output_path = Path('segmentation_output')
        self.output_path.mkdir(exist_ok=True)
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the customer data for clustering.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        df = self.df.copy()
        
        # Handle missing values
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
        
        # Scale numerical variables
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns
        )
        
        self.preprocessed_data = df_scaled
        return df_scaled
    
    def find_optimal_clusters(self, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            max_clusters (int): Maximum number of clusters to try
            
        Returns:
            int: Optimal number of clusters
        """
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(self.preprocessed_data)
            silhouette_avg = silhouette_score(self.preprocessed_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        return silhouette_scores.index(max(silhouette_scores)) + 2
    
    def perform_clustering(self, n_clusters: int = None) -> np.ndarray:
        """
        Perform customer segmentation using KMeans clustering.
        
        Args:
            n_clusters (int): Number of clusters (optional)
            
        Returns:
            np.ndarray: Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        # Perform KMeans clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        self.clusters = kmeans.fit_predict(self.preprocessed_data)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        self.pca_components = pca.fit_transform(self.preprocessed_data)
        
        return self.clusters
    
    def generate_profiles(self) -> Dict[str, Any]:
        """
        Generate customer profiles for each segment using AI.
        
        Returns:
            Dict[str, Any]: Segment profiles
        """
        profiles = {}
        
        for cluster_id in range(len(np.unique(self.clusters))):
            # Get cluster data
            cluster_data = self.df[self.clusters == cluster_id]
            
            # Calculate cluster statistics
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df) * 100,
                'numeric_means': cluster_data.select_dtypes(include=[np.number]).mean().to_dict(),
                'categorical_modes': cluster_data.select_dtypes(include=['object']).mode().iloc[0].to_dict()
            }
            
            # Generate profile using AI
            prompt = f"""
            Generate a customer segment profile based on these statistics:
            {json.dumps(stats, indent=2)}
            
            Please provide:
            1. Segment name/label
            2. Key characteristics
            3. Behavioral patterns
            4. Preferences and interests
            5. Recommended marketing strategies
            6. Potential growth opportunities
            
            Make the profile specific and actionable.
            Respond in JSON format with structured profile information.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a customer segmentation expert. Create insightful customer profiles."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                profile = json.loads(response.choices[0].message.content)
                profiles[f"Segment_{cluster_id}"] = {
                    'statistics': stats,
                    'profile': profile
                }
                
            except Exception as e:
                logging.error(f"Error generating profile for cluster {cluster_id}: {str(e)}")
        
        self.profiles = profiles
        return profiles
    
    def create_visualizations(self) -> None:
        """
        Create interactive visualizations of the customer segments.
        """
        st.title("Customer Segmentation Analysis")
        
        # 1. Cluster visualization using PCA
        st.header("Customer Segments Visualization")
        fig = px.scatter(
            x=self.pca_components[:, 0],
            y=self.pca_components[:, 1],
            color=self.clusters,
            title="Customer Segments (PCA)",
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
        )
        st.plotly_chart(fig)
        
        # 2. Segment sizes
        st.header("Segment Sizes")
        segment_sizes = pd.Series(self.clusters).value_counts()
        fig = px.pie(
            values=segment_sizes.values,
            names=segment_sizes.index,
            title="Customer Segment Distribution"
        )
        st.plotly_chart(fig)
        
        # 3. Feature distributions by segment
        st.header("Feature Distributions by Segment")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            fig = px.box(
                x=self.clusters,
                y=self.df[column],
                title=f"{column} Distribution by Segment"
            )
            st.plotly_chart(fig)
        
        # 4. Segment profiles
        st.header("Segment Profiles")
        for segment, details in self.profiles.items():
            with st.expander(f"{segment}: {details['profile'].get('name', 'Unnamed Segment')}"):
                # Display profile information
                st.subheader("Key Characteristics")
                for char in details['profile'].get('characteristics', []):
                    st.write(f"- {char}")
                
                st.subheader("Behavioral Patterns")
                for pattern in details['profile'].get('behavioral_patterns', []):
                    st.write(f"- {pattern}")
                
                st.subheader("Marketing Strategies")
                for strategy in details['profile'].get('marketing_strategies', []):
                    st.write(f"- {strategy}")
                
                # Display statistics
                st.subheader("Statistics")
                stats = details['statistics']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Size: {stats['size']} customers")
                    st.write(f"Percentage: {stats['percentage']:.1f}%")
                
                with col2:
                    st.write("Average Values:")
                    for metric, value in stats['numeric_means'].items():
                        st.write(f"- {metric}: {value:.2f}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive segmentation analysis report.
        
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_info': {
                'n_customers': len(self.df),
                'n_segments': len(self.profiles)
            },
            'segments': self.profiles,
            'feature_importance': {
                col: abs(np.corrcoef(self.df[col], self.clusters)[0, 1])
                for col in self.df.select_dtypes(include=[np.number]).columns
            }
        }
        
        prompt = f"""
        Generate a customer segmentation report based on these results:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Methodology
        3. Segment Profiles
        4. Key Insights
        5. Recommendations
        6. Next Steps
        
        Format the report in Markdown, including sections for visualizations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a customer segmentation expert. Create comprehensive, actionable reports."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'segmentation_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def save_results(self) -> None:
        """
        Save segmentation results to files.
        """
        # Save cluster assignments
        results_df = self.df.copy()
        results_df['Segment'] = self.clusters
        results_df.to_csv(self.output_path / 'segmentation_results.csv', index=False)
        
        # Save profiles
        with open(self.output_path / 'segment_profiles.json', 'w') as f:
            json.dump(self.profiles, f, indent=2)
        
        # Save PCA components
        pca_df = pd.DataFrame(
            self.pca_components,
            columns=['PC1', 'PC2']
        )
        pca_df['Segment'] = self.clusters
        pca_df.to_csv(self.output_path / 'pca_results.csv', index=False)

def main():
    """
    Main function to run the Customer Segmentation system.
    """
    try:
        st.title("AI-Powered Customer Segmentation")
        st.write("""
        This tool helps you segment your customers using advanced clustering algorithms
        and generates detailed profiles for each segment using AI.
        """)
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file with customer data", type="csv")
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Initialize segmentation
            segmentation = CustomerSegmentation(df)
            
            if st.button("Perform Segmentation"):
                with st.spinner("Analyzing customer segments..."):
                    # Preprocess data
                    segmentation.preprocess_data()
                    
                    # Perform clustering
                    n_clusters = st.slider("Number of segments", 2, 10, 5)
                    segmentation.perform_clustering(n_clusters)
                    
                    # Generate profiles
                    segmentation.generate_profiles()
                    
                    # Create visualizations
                    segmentation.create_visualizations()
                    
                    # Generate and display report
                    report = segmentation.generate_report()
                    st.markdown(report)
                    
                    # Save results
                    segmentation.save_results()
                    
                    # Download buttons
                    st.download_button(
                        "Download Segmentation Results",
                        df.to_csv(index=False),
                        "segmentation_results.csv",
                        "text/csv"
                    )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 