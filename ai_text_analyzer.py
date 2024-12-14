"""
AI-Powered Text Analysis Tool
This script performs comprehensive analysis of customer reviews using advanced NLP techniques.
It combines multiple AI models to:
- Extract sentiment and emotions
- Identify key topics and themes
- Categorize feedback
- Generate actionable insights
- Detect urgent issues
- Summarize feedback trends
- Extract product/service aspects

The script leverages:
- GPT-4 for advanced analysis and insight generation
- Hugging Face transformers for sentiment analysis
- spaCy for NLP tasks
- NLTK for text preprocessing
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import pipeline
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
OPENAI_API_KEY = "your_api_key_here"
client = OpenAI(api_key=OPENAI_API_KEY)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AITextAnalyzer:
    def __init__(self, reviews_df: pd.DataFrame):
        """
        Initialize the AI-powered Text Analyzer.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing customer reviews
                                     Expected columns: ['review_text', 'date']
                                     Optional columns: ['rating', 'customer_id', 'product_id']
        """
        self.df = reviews_df.copy()
        self.insights = []
        self.output_path = Path('text_analysis_output')
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize NLP models
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.summarizer = pipeline('summarization')
        
        # Set up visualization style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            str: Cleaned text
        """
        # Basic cleaning
        text = text.lower().strip()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Remove stopwords and punctuation
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self) -> Dict[str, Any]:
        """
        Perform detailed sentiment analysis on reviews.
        Combines transformer-based sentiment analysis with GPT-4 emotional analysis.
        
        Returns:
            Dict: Sentiment analysis results and insights
        """
        # Basic sentiment analysis with transformers
        sentiments = []
        for text in self.df['review_text']:
            try:
                result = self.sentiment_analyzer(text[:512])[0]  # Truncate to max length
                sentiments.append(result)
            except Exception as e:
                logging.error(f"Error in sentiment analysis: {str(e)}")
                sentiments.append({'label': 'UNKNOWN', 'score': 0.0})
        
        # Add sentiment results to DataFrame
        self.df['sentiment_label'] = [s['label'] for s in sentiments]
        self.df['sentiment_score'] = [s['score'] for s in sentiments]
        
        # Get deeper emotional analysis with GPT-4
        sample_reviews = self.df['review_text'].head(5).tolist()  # Analyze a sample for efficiency
        
        prompt = f"""
        Analyze the emotional content and tone of these customer reviews:
        {json.dumps(sample_reviews, indent=2)}
        
        Provide insights about:
        1. Dominant emotions expressed
        2. Common emotional patterns
        3. Customer satisfaction indicators
        4. Areas of emotional concern
        
        Respond in JSON format with structured insights.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in emotional analysis and customer psychology."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            emotional_insights = json.loads(response.choices[0].message.content)
            self.insights.append(('Emotional Analysis', emotional_insights))
            
        except Exception as e:
            logging.error(f"Error in emotional analysis: {str(e)}")
            emotional_insights = {"error": "Emotional analysis failed"}
        
        return {
            'sentiment_distribution': self.df['sentiment_label'].value_counts().to_dict(),
            'average_sentiment_score': float(self.df['sentiment_score'].mean()),
            'emotional_insights': emotional_insights
        }
    
    def extract_topics(self) -> Dict[str, Any]:
        """
        Extract main topics and themes from reviews using AI analysis.
        Combines keyword extraction with GPT-4 topic analysis.
        
        Returns:
            Dict: Extracted topics and related insights
        """
        # Extract key phrases with spaCy
        key_phrases = []
        for text in self.df['review_text']:
            doc = self.nlp(text)
            phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
            key_phrases.extend(phrases)
        
        # Get most common phrases
        phrase_counter = Counter(key_phrases)
        common_phrases = dict(phrase_counter.most_common(10))
        
        # Get AI interpretation of topics
        prompt = f"""
        Analyze these frequently mentioned phrases from customer reviews:
        {json.dumps(common_phrases, indent=2)}
        
        Also analyze these sample reviews for context:
        {json.dumps(self.df['review_text'].head(5).tolist(), indent=2)}
        
        Provide insights about:
        1. Main topics/themes discussed
        2. Common customer concerns
        3. Product/service aspects mentioned
        4. Suggested topic categories
        
        Respond in JSON format with structured insights.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in topic analysis and customer feedback categorization."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            topic_insights = json.loads(response.choices[0].message.content)
            self.insights.append(('Topic Analysis', topic_insights))
            
        except Exception as e:
            logging.error(f"Error in topic analysis: {str(e)}")
            topic_insights = {"error": "Topic analysis failed"}
        
        return {
            'common_phrases': common_phrases,
            'topic_insights': topic_insights
        }
    
    def categorize_feedback(self) -> Dict[str, Any]:
        """
        Categorize reviews into meaningful categories using AI analysis.
        Uses GPT-4 to create and assign categories based on content.
        
        Returns:
            Dict: Categorization results and insights
        """
        # Sample reviews for category development
        sample_reviews = self.df['review_text'].head(10).tolist()
        
        # Get AI to develop categories and categorize reviews
        prompt = f"""
        Analyze these customer reviews and develop a categorization system:
        {json.dumps(sample_reviews, indent=2)}
        
        Please:
        1. Create 5-7 meaningful categories for these reviews
        2. Explain the criteria for each category
        3. Categorize the sample reviews
        4. Provide insights about the distribution of feedback
        
        Respond in JSON format with structured categories and analysis.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in customer feedback analysis and categorization."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            categorization = json.loads(response.choices[0].message.content)
            self.insights.append(('Feedback Categorization', categorization))
            
        except Exception as e:
            logging.error(f"Error in feedback categorization: {str(e)}")
            categorization = {"error": "Categorization failed"}
        
        return categorization
    
    def generate_summaries(self) -> Dict[str, str]:
        """
        Generate AI-powered summaries of review themes and trends.
        Combines transformer-based summarization with GPT-4 analysis.
        
        Returns:
            Dict: Various types of summaries and insights
        """
        # Group reviews by sentiment for targeted summarization
        positive_reviews = self.df[self.df['sentiment_score'] > 0.6]['review_text'].tolist()
        negative_reviews = self.df[self.df['sentiment_score'] < 0.4]['review_text'].tolist()
        
        summaries = {}
        
        # Generate summaries for different segments
        for review_type, reviews in [('positive', positive_reviews), ('negative', negative_reviews)]:
            if reviews:
                combined_text = ' '.join(reviews[:5])  # Limit to 5 reviews for efficiency
                
                try:
                    # Get transformer-based summary
                    summary = self.summarizer(combined_text, max_length=150, min_length=50)[0]['summary_text']
                    
                    # Get GPT-4 analysis of the summary
                    prompt = f"""
                    Analyze this summary of {review_type} customer reviews:
                    {summary}
                    
                    Provide:
                    1. Key themes and patterns
                    2. Notable customer sentiments
                    3. Actionable insights
                    4. Recommendations based on the feedback
                    
                    Respond in JSON format.
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert in customer feedback analysis."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    analysis = json.loads(response.choices[0].message.content)
                    summaries[f'{review_type}_feedback'] = {
                        'summary': summary,
                        'analysis': analysis
                    }
                    
                except Exception as e:
                    logging.error(f"Error in generating {review_type} summary: {str(e)}")
                    summaries[f'{review_type}_feedback'] = {"error": f"Summary generation failed for {review_type} reviews"}
        
        return summaries
    
    def visualize_insights(self) -> None:
        """
        Create visualizations of the analysis results.
        """
        # 1. Sentiment Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='sentiment_label')
        plt.title('Distribution of Review Sentiments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_path / 'sentiment_distribution.png')
        plt.close()
        
        # 2. Sentiment Score Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='sentiment_score', bins=20)
        plt.title('Distribution of Sentiment Scores')
        plt.tight_layout()
        plt.savefig(self.output_path / 'sentiment_scores.png')
        plt.close()
        
        # 3. Topic Frequency (if available)
        if hasattr(self, 'topic_insights'):
            topics = pd.DataFrame(self.topic_insights['common_phrases'].items(), 
                                columns=['Topic', 'Frequency'])
            plt.figure(figsize=(12, 6))
            sns.barplot(data=topics, x='Frequency', y='Topic')
            plt.title('Most Common Topics in Reviews')
            plt.tight_layout()
            plt.savefig(self.output_path / 'topic_frequency.png')
            plt.close()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report using AI.
        
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_info': {
                'total_reviews': len(self.df),
                'date_range': f"{self.df['date'].min()} to {self.df['date'].max()}"
            },
            'insights': self.insights
        }
        
        prompt = f"""
        Generate a comprehensive text analysis report based on these insights:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Key Findings
        3. Sentiment Analysis Results
        4. Topic Analysis
        5. Customer Feedback Categories
        6. Actionable Recommendations
        7. Areas for Improvement
        
        Format the report in Markdown.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a customer insights analyst. Create a comprehensive analysis report."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'analysis_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error in report generation: {str(e)}")
            return "Error generating report"
    
    def run_complete_analysis(self) -> Tuple[str, Path]:
        """
        Run the complete text analysis process.
        
        Returns:
            Tuple[str, Path]: Analysis report and path to output directory
        """
        logging.info("Starting AI-powered text analysis...")
        
        # Run all analyses
        sentiment_results = self.analyze_sentiment()
        topic_results = self.extract_topics()
        categorization_results = self.categorize_feedback()
        summary_results = self.generate_summaries()
        
        # Create visualizations
        self.visualize_insights()
        
        # Generate final report
        report = self.generate_report()
        
        logging.info(f"Analysis complete. Output saved to {self.output_path}")
        return report, self.output_path

def main():
    """
    Example usage of the AITextAnalyzer class
    """
    try:
        # Load sample customer reviews
        reviews_df = pd.read_csv('reviews.csv')
        
        print("Starting AI-Powered Text Analysis...")
        analyzer = AITextAnalyzer(reviews_df)
        report, output_path = analyzer.run_complete_analysis()
        
        print(f"\nAnalysis complete! Check the following locations for results:")
        print(f"- Report: {output_path / 'analysis_report.md'}")
        print(f"- Visualizations: {output_path}")
        print("\nReport Preview:")
        print("="*50)
        print(report[:500] + "...\n")
        
    except FileNotFoundError:
        print("Error: reviews.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 