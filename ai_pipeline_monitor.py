"""
Real-Time Data Pipeline Monitoring Tutorial
=======================================

This script demonstrates how to build an intelligent monitoring system for data
pipelines that uses AI to detect anomalies, delays, and data quality issues.

Features:
1. Real-time pipeline monitoring
2. AI-powered anomaly detection
3. Data quality checks
4. Performance metrics tracking
5. Smart alerting system
6. Historical analysis
7. Dashboard visualization

Author: [Your Name]
Date: [Current Date]
License: MIT
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, count, avg, stddev
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from openai import OpenAI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import requests
import time
import threading
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PipelineMonitor:
    """
    A class that monitors data pipelines and provides real-time insights and alerts.
    """
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        """
        Initialize the Pipeline Monitor.
        
        Args:
            pipeline_config (Dict[str, Any]): Configuration for pipeline monitoring
        """
        self.config = pipeline_config
        self.metrics = {}
        self.alerts = Queue()
        self.is_monitoring = False
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("PipelineMonitor") \
            .getOrCreate()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Create output directory
        self.output_path = Path('pipeline_output')
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize monitoring thread
        self.monitor_thread = None
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks on incoming data.
        
        Args:
            df (pd.DataFrame): Input dataframe to check
            
        Returns:
            Dict[str, Any]: Data quality metrics and issues
        """
        quality_metrics = {
            'row_count': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'column_stats': {}
        }
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                quality_metrics['column_stats'][column] = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max())
                }
            elif pd.api.types.is_string_dtype(df[column]):
                quality_metrics['column_stats'][column] = {
                    'unique_count': df[column].nunique(),
                    'most_common': df[column].value_counts().head(5).to_dict()
                }
        
        return quality_metrics
    
    def detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in pipeline metrics using AI.
        
        Args:
            metrics (Dict[str, Any]): Pipeline metrics to analyze
            
        Returns:
            List[Dict[str, Any]]: Detected anomalies
        """
        # Prepare metrics for analysis
        analysis_context = {
            'current_metrics': metrics,
            'historical_metrics': self.metrics,
            'thresholds': self.config.get('thresholds', {})
        }
        
        prompt = f"""
        Analyze these pipeline metrics for anomalies:
        {json.dumps(analysis_context, indent=2)}
        
        Please identify:
        1. Significant deviations from normal patterns
        2. Performance bottlenecks
        3. Data quality issues
        4. System health concerns
        5. Potential root causes
        
        Focus on critical issues that require immediate attention.
        Respond in JSON format with structured anomaly information.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data pipeline expert. Identify critical issues and anomalies."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            anomalies = json.loads(response.choices[0].message.content)
            return anomalies.get('anomalies', [])
            
        except Exception as e:
            logging.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def generate_alert(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an alert based on detected anomaly.
        
        Args:
            anomaly (Dict[str, Any]): Detected anomaly information
            
        Returns:
            Dict[str, Any]: Alert details
        """
        prompt = f"""
        Generate an alert for this pipeline anomaly:
        {json.dumps(anomaly, indent=2)}
        
        Please provide:
        1. Alert severity level
        2. Clear description of the issue
        3. Potential impact
        4. Recommended actions
        5. Additional context
        
        Format the alert for technical teams.
        Respond in JSON format with structured alert information.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a pipeline monitoring expert. Generate clear, actionable alerts."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            alert = json.loads(response.choices[0].message.content)
            alert['timestamp'] = datetime.now().isoformat()
            return alert
            
        except Exception as e:
            logging.error(f"Error generating alert: {str(e)}")
            return {
                "error": "Alert generation failed",
                "details": str(e)
            }
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert through configured channels (e.g., Slack, email).
        
        Args:
            alert (Dict[str, Any]): Alert to send
            
        Returns:
            bool: Whether alert was sent successfully
        """
        # Example: Send to Slack
        if 'slack_webhook' in self.config:
            try:
                message = {
                    "text": f"🚨 Pipeline Alert\n"
                           f"Severity: {alert.get('severity', 'Unknown')}\n"
                           f"Issue: {alert.get('description', 'No description')}\n"
                           f"Impact: {alert.get('impact', 'Unknown')}\n"
                           f"Action: {alert.get('recommended_action', 'No action specified')}"
                }
                
                response = requests.post(
                    self.config['slack_webhook'],
                    json=message
                )
                
                return response.status_code == 200
                
            except Exception as e:
                logging.error(f"Error sending Slack alert: {str(e)}")
                return False
        
        return False
    
    def monitor_pipeline(self) -> None:
        """
        Main monitoring loop for the pipeline.
        """
        while self.is_monitoring:
            try:
                # 1. Collect current metrics
                current_metrics = self.collect_metrics()
                
                # 2. Check data quality
                quality_metrics = self.check_data_quality(
                    pd.DataFrame(current_metrics['recent_data'])
                )
                
                # 3. Detect anomalies
                anomalies = self.detect_anomalies({
                    **current_metrics,
                    'quality_metrics': quality_metrics
                })
                
                # 4. Generate and send alerts for anomalies
                for anomaly in anomalies:
                    alert = self.generate_alert(anomaly)
                    if alert and 'error' not in alert:
                        self.alerts.put(alert)
                        self.send_alert(alert)
                
                # 5. Update historical metrics
                self.metrics[datetime.now().isoformat()] = {
                    **current_metrics,
                    'quality_metrics': quality_metrics
                }
                
                # 6. Clean up old metrics
                self.cleanup_old_metrics()
                
                # Wait for next monitoring interval
                time.sleep(self.config.get('monitoring_interval', 60))
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current pipeline metrics.
        
        Returns:
            Dict[str, Any]: Current pipeline metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {},
            'system_metrics': {},
            'recent_data': []
        }
        
        try:
            # Example: Collect Spark metrics
            spark_metrics = self.spark.sparkContext.statusTracker().getExecutorMetrics()
            metrics['performance_metrics']['spark'] = spark_metrics
            
            # Example: Collect system metrics
            # Add your system monitoring code here
            
            # Example: Get recent data sample
            if 'input_path' in self.config:
                recent_data = self.spark.read.parquet(self.config['input_path']) \
                    .orderBy(col('timestamp').desc()) \
                    .limit(1000) \
                    .toPandas()
                metrics['recent_data'] = recent_data.to_dict('records')
            
        except Exception as e:
            logging.error(f"Error collecting metrics: {str(e)}")
        
        return metrics
    
    def cleanup_old_metrics(self) -> None:
        """
        Remove metrics older than retention period.
        """
        retention_hours = self.config.get('metrics_retention_hours', 24)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        self.metrics = {
            timestamp: metrics
            for timestamp, metrics in self.metrics.items()
            if datetime.fromisoformat(timestamp) > cutoff_time
        }
    
    def start_monitoring(self) -> None:
        """
        Start the pipeline monitoring process.
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_pipeline)
            self.monitor_thread.start()
            logging.info("Pipeline monitoring started")
    
    def stop_monitoring(self) -> None:
        """
        Stop the pipeline monitoring process.
        """
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
            logging.info("Pipeline monitoring stopped")
    
    def create_dashboard(self) -> None:
        """
        Create an interactive monitoring dashboard using Streamlit.
        """
        st.title("Real-Time Pipeline Monitor")
        
        # Sidebar controls
        st.sidebar.title("Controls")
        if not self.is_monitoring:
            if st.sidebar.button("Start Monitoring"):
                self.start_monitoring()
        else:
            if st.sidebar.button("Stop Monitoring"):
                self.stop_monitoring()
        
        # Main dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Pipeline Status")
            if self.is_monitoring:
                st.success("🟢 Pipeline Monitoring Active")
            else:
                st.error("🔴 Pipeline Monitoring Inactive")
        
        with col2:
            st.header("Recent Alerts")
            alerts_list = list(self.alerts.queue)
            for alert in alerts_list[-5:]:  # Show last 5 alerts
                severity = alert.get('severity', 'Unknown')
                color = {
                    'Critical': 'red',
                    'High': 'orange',
                    'Medium': 'yellow',
                    'Low': 'blue'
                }.get(severity, 'grey')
                
                st.markdown(f"""
                <div style='padding: 10px; border-left: 5px solid {color}; margin: 5px 0;'>
                    <strong>{severity}</strong>: {alert.get('description', 'No description')}
                    <br><small>{alert.get('timestamp', '')}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Metrics visualization
        st.header("Pipeline Metrics")
        if self.metrics:
            # Convert metrics to DataFrame for visualization
            metrics_df = pd.DataFrame([
                {
                    'timestamp': timestamp,
                    **{f"metric_{k}": v for k, v in metrics.items() if k != 'timestamp'}
                }
                for timestamp, metrics in self.metrics.items()
            ])
            
            # Plot metrics over time
            fig = px.line(
                metrics_df,
                x='timestamp',
                y=[col for col in metrics_df.columns if col.startswith('metric_')],
                title="Pipeline Metrics Over Time"
            )
            st.plotly_chart(fig)
        
        # Data quality metrics
        st.header("Data Quality")
        if self.metrics:
            latest_metrics = list(self.metrics.values())[-1]
            quality_metrics = latest_metrics.get('quality_metrics', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Row Statistics")
                st.write(f"Total Rows: {quality_metrics.get('row_count', 0)}")
                st.write(f"Duplicates: {quality_metrics.get('duplicates', 0)}")
            
            with col2:
                st.subheader("Null Values")
                null_counts = quality_metrics.get('null_counts', {})
                for col, count in null_counts.items():
                    st.write(f"{col}: {count}")
        
        # System health
        st.header("System Health")
        if self.metrics:
            latest_metrics = list(self.metrics.values())[-1]
            system_metrics = latest_metrics.get('system_metrics', {})
            
            # Example: Show Spark executor metrics
            if 'spark' in system_metrics:
                st.subheader("Spark Metrics")
                spark_metrics = system_metrics['spark']
                st.write(spark_metrics)

def main():
    """
    Main function to run the Pipeline Monitor.
    """
    try:
        # Example configuration
        config = {
            'monitoring_interval': 60,  # seconds
            'metrics_retention_hours': 24,
            'thresholds': {
                'latency_ms': 1000,
                'error_rate': 0.01,
                'data_quality_score': 0.95
            },
            'slack_webhook': st.secrets.get("SLACK_WEBHOOK_URL"),
            'input_path': 'path/to/pipeline/data'  # Update with your data path
        }
        
        # Initialize and run monitor
        monitor = PipelineMonitor(config)
        monitor.create_dashboard()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 