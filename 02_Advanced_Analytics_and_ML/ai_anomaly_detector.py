"""
AI-Powered Anomaly Detection Tutorial
==================================

This script demonstrates how to use various AI techniques to detect anomalies
in time series data. It combines traditional statistical methods with
modern AI approaches for robust anomaly detection.

Methods demonstrated:
1. Isolation Forest for outlier detection
2. Autoencoder for pattern-based anomalies
3. GPT-4 for contextual analysis
4. Statistical methods (Z-score, IQR)
5. Time series decomposition
6. Ensemble approach

Author: Sujal Dhungana , Manish Paneru
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from openai import OpenAI
import json
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
OPENAI_API_KEY = "your_api_key_here"
client = OpenAI(api_key=OPENAI_API_KEY)

class AIAnomalyDetector:
    """
    A class that combines multiple AI techniques for anomaly detection.
    It uses an ensemble approach to identify different types of anomalies.
    """
    
    def __init__(self, df: pd.DataFrame, timestamp_col: str, value_col: str):
        """
        Initialize the Anomaly Detector.
        
        Args:
            df (pd.DataFrame): Input dataset
            timestamp_col (str): Name of timestamp column
            value_col (str): Name of value column to analyze
        """
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.scaler = StandardScaler()
        self.models = {}
        self.anomalies = {}
        
        # Ensure timestamp is datetime
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        
        # Sort by timestamp
        self.df = self.df.sort_values(timestamp_col)
        
        # Create output directory
        self.output_path = Path('anomaly_output')
        self.output_path.mkdir(exist_ok=True)
    
    def preprocess_data(self) -> np.ndarray:
        """
        Preprocess the data for anomaly detection.
        
        Returns:
            np.ndarray: Preprocessed data
        """
        # Scale the data
        values = self.df[self.value_col].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        return scaled_values
    
    def detect_statistical_anomalies(self) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using statistical methods (Z-score and IQR).
        
        Returns:
            Dict: Anomaly indices for each method
        """
        values = self.df[self.value_col].values
        
        # Z-score method
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        z_score_anomalies = z_scores > 3
        
        # IQR method
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        iqr_anomalies = (values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))
        
        self.anomalies['statistical'] = {
            'z_score': z_score_anomalies,
            'iqr': iqr_anomalies
        }
        
        return self.anomalies['statistical']
    
    def detect_isolation_forest_anomalies(self) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Returns:
            np.ndarray: Boolean array indicating anomalies
        """
        # Prepare features
        scaled_values = self.preprocess_data()
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(scaled_values)
        
        # Convert predictions to boolean array (True for anomalies)
        anomalies = predictions == -1
        
        self.models['isolation_forest'] = iso_forest
        self.anomalies['isolation_forest'] = anomalies
        
        return anomalies
    
    def build_autoencoder(self) -> Model:
        """
        Build and compile an autoencoder model for anomaly detection.
        
        Returns:
            Model: Compiled autoencoder model
        """
        # Define model architecture
        input_dim = 1
        encoding_dim = 32
        
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoder = Dense(encoding_dim, activation='relu')(input_layer)
        encoder = Dense(16, activation='relu')(encoder)
        encoder = Dense(8, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(16, activation='relu')(encoder)
        decoder = Dense(encoding_dim, activation='relu')(decoder)
        decoder = Dense(input_dim, activation='linear')(decoder)
        
        # Create model
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def detect_autoencoder_anomalies(self, threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect anomalies using an autoencoder.
        
        Args:
            threshold_percentile (float): Percentile for reconstruction error threshold
            
        Returns:
            np.ndarray: Boolean array indicating anomalies
        """
        # Prepare data
        scaled_values = self.preprocess_data()
        
        # Build and train autoencoder
        autoencoder = self.build_autoencoder()
        autoencoder.fit(scaled_values, scaled_values, 
                       epochs=50, batch_size=32, shuffle=True, verbose=0)
        
        # Get reconstruction error
        reconstructed = autoencoder.predict(scaled_values)
        reconstruction_error = np.mean(np.abs(scaled_values - reconstructed), axis=1)
        
        # Determine threshold and identify anomalies
        threshold = np.percentile(reconstruction_error, threshold_percentile)
        anomalies = reconstruction_error > threshold
        
        self.models['autoencoder'] = autoencoder
        self.anomalies['autoencoder'] = anomalies
        
        return anomalies
    
    def analyze_with_gpt(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """
        Use GPT-4 to analyze potential anomalies in context.
        
        Args:
            window_size (int): Size of context window around anomalies
            
        Returns:
            List[Dict]: List of anomaly analyses
        """
        # Combine anomalies from different methods
        combined_anomalies = np.zeros_like(self.df[self.value_col].values, dtype=bool)
        for method_anomalies in self.anomalies.values():
            if isinstance(method_anomalies, dict):
                for anomaly in method_anomalies.values():
                    combined_anomalies |= anomaly
            else:
                combined_anomalies |= method_anomalies
        
        # Get indices of anomalies
        anomaly_indices = np.where(combined_anomalies)[0]
        
        analyses = []
        for idx in anomaly_indices:
            # Get context window
            start_idx = max(0, idx - window_size)
            end_idx = min(len(self.df), idx + window_size)
            
            context = {
                'timestamp': self.df[self.timestamp_col].iloc[idx],
                'value': self.df[self.value_col].iloc[idx],
                'window_data': self.df.iloc[start_idx:end_idx][[self.timestamp_col, self.value_col]].to_dict(),
                'detection_methods': [method for method, anomalies in self.anomalies.items() 
                                   if (isinstance(anomalies, np.ndarray) and anomalies[idx]) or
                                   (isinstance(anomalies, dict) and any(a[idx] for a in anomalies.values()))]
            }
            
            prompt = f"""
            Analyze this potential anomaly in the time series data:
            {json.dumps(context, default=str, indent=2)}
            
            Please provide:
            1. Assessment of whether this is a true anomaly
            2. Possible causes or explanations
            3. Severity level
            4. Recommended actions
            
            Respond in JSON format with structured analysis.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert in time series analysis and anomaly detection."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                analysis = json.loads(response.choices[0].message.content)
                analysis['index'] = idx
                analyses.append(analysis)
                
            except Exception as e:
                logging.error(f"Error analyzing anomaly at index {idx}: {str(e)}")
        
        return analyses
    
    def visualize_results(self) -> None:
        """
        Create visualizations of the anomaly detection results.
        """
        # 1. Time series plot with anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(self.df[self.timestamp_col], self.df[self.value_col], label='Original')
        
        # Plot anomalies from each method
        colors = ['red', 'orange', 'green', 'purple']
        for (method, anomalies), color in zip(self.anomalies.items(), colors):
            if isinstance(anomalies, dict):
                for anomaly_type, mask in anomalies.items():
                    plt.scatter(self.df[self.timestamp_col][mask],
                              self.df[self.value_col][mask],
                              c=color, label=f'{method}-{anomaly_type}', alpha=0.5)
            else:
                plt.scatter(self.df[self.timestamp_col][anomalies],
                          self.df[self.value_col][anomalies],
                          c=color, label=method, alpha=0.5)
        
        plt.title('Time Series with Detected Anomalies')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_path / 'anomalies_timeseries.png')
        plt.close()
        
        # 2. Decomposition plot
        decomposition = seasonal_decompose(self.df[self.value_col], period=30)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'time_series_decomposition.png')
        plt.close()
        
        # 3. Distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=self.value_col, bins=50)
        plt.title('Value Distribution')
        plt.savefig(self.output_path / 'value_distribution.png')
        plt.close()
    
    def generate_report(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive anomaly detection report.
        
        Args:
            analyses (List[Dict]): List of GPT-4 analyses of anomalies
            
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_info': {
                'n_samples': len(self.df),
                'time_range': f"{self.df[self.timestamp_col].min()} to {self.df[self.timestamp_col].max()}",
                'value_range': f"{self.df[self.value_col].min():.2f} to {self.df[self.value_col].max():.2f}"
            },
            'anomaly_counts': {
                method: np.sum(anomalies) if isinstance(anomalies, np.ndarray)
                else {k: np.sum(v) for k, v in anomalies.items()}
                for method, anomalies in self.anomalies.items()
            },
            'analyses': analyses
        }
        
        prompt = f"""
        Generate a comprehensive anomaly detection report based on these results:
        {json.dumps(report_context, default=str, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Methodology Overview
        3. Key Findings
        4. Detailed Anomaly Analysis
        5. Patterns and Trends
        6. Recommendations
        
        Format the report in Markdown, including sections for visualizations.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in anomaly detection and time series analysis."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'anomaly_detection_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def run_complete_analysis(self) -> Tuple[str, Path]:
        """
        Run the complete anomaly detection pipeline.
        
        Returns:
            Tuple[str, Path]: Analysis report and path to output directory
        """
        logging.info("Starting anomaly detection pipeline...")
        
        # 1. Statistical anomaly detection
        logging.info("Detecting statistical anomalies...")
        self.detect_statistical_anomalies()
        
        # 2. Isolation Forest
        logging.info("Running Isolation Forest...")
        self.detect_isolation_forest_anomalies()
        
        # 3. Autoencoder
        logging.info("Training autoencoder...")
        self.detect_autoencoder_anomalies()
        
        # 4. GPT-4 analysis
        logging.info("Analyzing anomalies with GPT-4...")
        analyses = self.analyze_with_gpt()
        
        # 5. Create visualizations
        logging.info("Creating visualizations...")
        self.visualize_results()
        
        # 6. Generate report
        logging.info("Generating final report...")
        report = self.generate_report(analyses)
        
        logging.info(f"Analysis complete. Output saved to {self.output_path}")
        return report, self.output_path

def main():
    """
    Example usage of the AIAnomalyDetector class.
    """
    try:
        # Load sample dataset
        df = pd.read_csv('data.csv')
        
        print("Starting AI-Powered Anomaly Detection...")
        detector = AIAnomalyDetector(df, timestamp_col='timestamp', value_col='value')
        report, output_path = detector.run_complete_analysis()
        
        print(f"\nAnalysis complete! Check the following locations for results:")
        print(f"- Report: {output_path / 'anomaly_detection_report.md'}")
        print(f"- Visualizations: {output_path}")
        print("\nReport Preview:")
        print("="*50)
        print(report[:500] + "...\n")
        
    except FileNotFoundError:
        print("Error: data.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 