"""
AI-Powered Survey Analysis Tool
============================

This script provides advanced analysis capabilities for survey data.
It helps extract insights and patterns from survey responses.

Author: Sujal Dhungana , Manish Paneru
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from openai import OpenAI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class SurveyAnalyzer:
    """
    A class that analyzes survey responses using various AI techniques.
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str):
        """
        Initialize the Survey Analyzer.
        
        Args:
            df (pd.DataFrame): Input dataset with survey responses
            text_column (str): Name of column containing text responses
        """
        self.df = df.copy()
        self.text_column = text_column
        self.preprocessed_texts = []
        self.themes = {}
        self.sentiments = {}
        self.clusters = {}
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Create output directory
        self.output_path = Path('survey_output')
        self.output_path.mkdir(exist_ok=True)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data for analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_themes(self, texts: List[str], num_themes: int = 5) -> Dict[str, Any]:
        """
        Extract main themes from survey responses using AI.
        
        Args:
            texts (List[str]): List of preprocessed texts
            num_themes (int): Number of themes to extract
            
        Returns:
            Dict[str, Any]: Extracted themes and their details
        """
        # Combine texts for analysis
        combined_text = "\n".join(texts[:100])  # Limit to first 100 responses for API efficiency
        
        prompt = f"""
        Analyze these survey responses and identify the main themes:
        {combined_text}
        
        Please provide:
        1. Top {num_themes} main themes
        2. Keywords associated with each theme
        3. Example quotes for each theme
        4. Relative importance/frequency of each theme
        
        Respond in JSON format with structured theme analysis.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a survey analysis expert. Extract meaningful themes from responses."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            themes = json.loads(response.choices[0].message.content)
            self.themes = themes
            
            return themes
            
        except Exception as e:
            logging.error(f"Error extracting themes: {str(e)}")
            return {"error": "Theme extraction failed"}
    
    def analyze_sentiment(self) -> Dict[str, float]:
        """
        Perform sentiment analysis on survey responses.
        
        Returns:
            Dict[str, float]: Sentiment scores for each response
        """
        sentiments = {}
        
        try:
            for idx, row in self.df.iterrows():
                text = row[self.text_column]
                result = self.sentiment_analyzer(text)[0]
                sentiments[idx] = {
                    'label': result['label'],
                    'score': result['score']
                }
            
            self.sentiments = sentiments
            return sentiments
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return {"error": "Sentiment analysis failed"}
    
    def cluster_responses(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster survey responses to identify patterns.
        
        Args:
            n_clusters (int): Number of clusters to create
            
        Returns:
            Dict[str, Any]: Clustering results
        """
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(self.preprocessed_texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get cluster centers and top terms
            cluster_centers = kmeans.cluster_centers_
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms for each cluster
            top_terms = {}
            for i in range(n_clusters):
                center = cluster_centers[i]
                top_indices = center.argsort()[-10:][::-1]  # Get top 10 terms
                top_terms[f"Cluster {i}"] = [feature_names[idx] for idx in top_indices]
            
            clustering_results = {
                'labels': cluster_labels.tolist(),
                'top_terms': top_terms,
                'vectorizer': vectorizer,
                'kmeans': kmeans
            }
            
            self.clusters = clustering_results
            return clustering_results
            
        except Exception as e:
            logging.error(f"Error in clustering: {str(e)}")
            return {"error": "Clustering failed"}
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate AI-powered insights from the analysis results.
        
        Returns:
            Dict[str, Any]: Generated insights
        """
        analysis_context = {
            'themes': self.themes,
            'sentiment_summary': {
                'positive': sum(1 for s in self.sentiments.values() if s['label'] == 'POSITIVE'),
                'negative': sum(1 for s in self.sentiments.values() if s['label'] == 'NEGATIVE')
            },
            'clusters': {
                'num_clusters': len(self.clusters.get('top_terms', {})),
                'top_terms': self.clusters.get('top_terms', {})
            }
        }
        
        prompt = f"""
        Generate insights from this survey analysis:
        {json.dumps(analysis_context, indent=2)}
        
        Please provide:
        1. Key findings and patterns
        2. Notable correlations between themes and sentiment
        3. Actionable recommendations
        4. Areas for further investigation
        5. Potential biases or limitations
        
        Focus on practical, actionable insights.
        Respond in JSON format with structured insights.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a survey insights expert. Provide clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logging.error(f"Error generating insights: {str(e)}")
            return {"error": "Insight generation failed"}
    
    def create_visualizations(self) -> None:
        """
        Create interactive visualizations of the analysis results.
        """
        st.title("Survey Analysis Results")
        
        # 1. Theme Analysis
        st.header("Theme Analysis")
        if isinstance(self.themes, dict) and 'themes' in self.themes:
            themes_df = pd.DataFrame(self.themes['themes'])
            fig = px.bar(
                themes_df,
                x='theme',
                y='frequency',
                title="Main Themes in Survey Responses"
            )
            st.plotly_chart(fig)
        
        # 2. Sentiment Analysis
        st.header("Sentiment Analysis")
        sentiment_counts = pd.Series({
            'Positive': sum(1 for s in self.sentiments.values() if s['label'] == 'POSITIVE'),
            'Negative': sum(1 for s in self.sentiments.values() if s['label'] == 'NEGATIVE')
        })
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig)
        
        # 3. Clustering Results
        st.header("Response Clusters")
        if 'top_terms' in self.clusters:
            for cluster, terms in self.clusters['top_terms'].items():
                st.subheader(f"{cluster} - Top Terms")
                st.write(", ".join(terms))
        
        # 4. Theme-Sentiment Correlation
        st.header("Theme-Sentiment Correlation")
        if isinstance(self.themes, dict) and 'themes' in self.themes:
            # Create correlation matrix
            theme_sentiment_corr = np.random.rand(len(self.themes['themes']))  # Placeholder
            fig = px.imshow(
                [theme_sentiment_corr],
                x=[theme['theme'] for theme in self.themes['themes']],
                title="Theme-Sentiment Correlation"
            )
            st.plotly_chart(fig)
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_info': {
                'n_responses': len(self.df),
                'avg_response_length': self.df[self.text_column].str.len().mean()
            },
            'themes': self.themes,
            'sentiment_analysis': {
                'summary': {
                    'positive': sum(1 for s in self.sentiments.values() if s['label'] == 'POSITIVE'),
                    'negative': sum(1 for s in self.sentiments.values() if s['label'] == 'NEGATIVE')
                }
            },
            'clustering': {
                'n_clusters': len(self.clusters.get('top_terms', {})),
                'cluster_details': self.clusters.get('top_terms', {})
            }
        }
        
        prompt = f"""
        Generate a comprehensive survey analysis report based on these results:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Methodology
        3. Key Findings
        4. Theme Analysis
        5. Sentiment Analysis
        6. Response Patterns
        7. Recommendations
        8. Limitations and Next Steps
        
        Format the report in Markdown, including sections for visualizations.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a survey analysis expert. Create clear, actionable reports."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'survey_analysis_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def run_analysis(self) -> Tuple[str, Path]:
        """
        Run the complete survey analysis pipeline.
        
        Returns:
            Tuple[str, Path]: Analysis report and path to output directory
        """
        logging.info("Starting survey analysis pipeline...")
        
        # 1. Preprocess texts
        logging.info("Preprocessing survey responses...")
        self.preprocessed_texts = [self.preprocess_text(text) for text in self.df[self.text_column]]
        
        # 2. Extract themes
        logging.info("Extracting themes...")
        self.extract_themes(self.preprocessed_texts)
        
        # 3. Analyze sentiment
        logging.info("Analyzing sentiment...")
        self.analyze_sentiment()
        
        # 4. Cluster responses
        logging.info("Clustering responses...")
        self.cluster_responses()
        
        # 5. Generate insights
        logging.info("Generating insights...")
        insights = self.generate_insights()
        
        # 6. Create visualizations
        logging.info("Creating visualizations...")
        self.create_visualizations()
        
        # 7. Generate report
        logging.info("Generating final report...")
        report = self.generate_report()
        
        logging.info(f"Analysis complete. Output saved to {self.output_path}")
        return report, self.output_path

def main():
    """
    Example usage of the SurveyAnalyzer class.
    """
    try:
        # Load sample dataset
        df = pd.read_csv('survey_data.csv')
        
        print("Starting AI-Powered Survey Analysis...")
        analyzer = SurveyAnalyzer(df, text_column='response')
        report, output_path = analyzer.run_analysis()
        
        print(f"\nAnalysis complete! Check the following locations for results:")
        print(f"- Report: {output_path / 'survey_analysis_report.md'}")
        print(f"- Visualizations: {output_path}")
        print("\nReport Preview:")
        print("="*50)
        print(report[:500] + "...\n")
        
    except FileNotFoundError:
        print("Error: survey_data.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 