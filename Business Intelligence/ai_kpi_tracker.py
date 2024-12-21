"""
AI-Powered KPI Tracking Tool
=========================

This script provides intelligent tracking and analysis of Key Performance Indicators.
It helps monitor and analyze KPIs using advanced AI techniques.

Author: Sujal Dhungana , Manish Paneru
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from prophet import Prophet
import json
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class KPITracker:
    """
    A class that tracks KPIs and provides AI-powered insights and recommendations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the KPI Tracker.
        
        Args:
            df (pd.DataFrame): Input dataset containing KPI metrics
        """
        self.df = df.copy()
        self.insights = []
        self.forecasts = {}
        self.recommendations = {}
        
        # Create output directory
        self.output_path = Path('kpi_output')
        self.output_path.mkdir(exist_ok=True)
    
    def calculate_kpis(self) -> Dict[str, pd.Series]:
        """
        Calculate various KPIs from the dataset.
        Customize this method based on your specific KPIs.
        
        Returns:
            Dict[str, pd.Series]: Dictionary of calculated KPIs
        """
        kpis = {}
        
        # Example KPI calculations (customize based on your needs)
        if 'sales' in self.df.columns and 'date' in self.df.columns:
            # Daily sales
            kpis['daily_sales'] = self.df.groupby('date')['sales'].sum()
            
            # 7-day moving average
            kpis['sales_ma_7d'] = kpis['daily_sales'].rolling(7).mean()
            
            # Month-over-month growth
            monthly_sales = self.df.groupby(pd.Grouper(key='date', freq='M'))['sales'].sum()
            kpis['mom_growth'] = monthly_sales.pct_change() * 100
        
        if 'customers' in self.df.columns:
            # Daily active customers
            kpis['daily_active_customers'] = self.df.groupby('date')['customers'].nunique()
            
            # Customer retention (example)
            def calculate_retention(df):
                total_customers = df['customers'].nunique()
                returning_customers = df[df['customers'].duplicated()]['customers'].nunique()
                return (returning_customers / total_customers) * 100 if total_customers > 0 else 0
            
            kpis['retention_rate'] = self.df.groupby(pd.Grouper(key='date', freq='M')).apply(calculate_retention)
        
        return kpis
    
    def analyze_trends(self, kpis: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Analyze trends in KPIs using AI.
        
        Args:
            kpis (Dict[str, pd.Series]): Dictionary of calculated KPIs
            
        Returns:
            Dict[str, Any]: Trend analysis results
        """
        # Prepare KPI statistics for analysis
        kpi_stats = {}
        for kpi_name, kpi_values in kpis.items():
            kpi_stats[kpi_name] = {
                'current_value': float(kpi_values.iloc[-1]),
                'previous_value': float(kpi_values.iloc[-2]),
                'change_pct': float(((kpi_values.iloc[-1] - kpi_values.iloc[-2]) / kpi_values.iloc[-2]) * 100),
                'mean': float(kpi_values.mean()),
                'std': float(kpi_values.std()),
                'trend_7d': kpi_values.tail(7).tolist()
            }
        
        prompt = f"""
        Analyze these KPI trends and provide insights:
        {json.dumps(kpi_stats, indent=2)}
        
        Please provide:
        1. Key trends and patterns
        2. Areas of concern
        3. Notable achievements
        4. Potential factors affecting performance
        5. Short-term outlook
        
        Focus on actionable insights that can help improve performance.
        Respond in JSON format with structured analysis.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a KPI analysis expert. Provide clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = json.loads(response.choices[0].message.content)
            self.insights.append(('Trend Analysis', analysis))
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in trend analysis: {str(e)}")
            return {"error": "Trend analysis failed"}
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI-powered recommendations based on KPI analysis.
        
        Args:
            analysis (Dict[str, Any]): Results from trend analysis
            
        Returns:
            Dict[str, Any]: Recommendations and action items
        """
        prompt = f"""
        Based on this KPI analysis, provide specific recommendations:
        {json.dumps(analysis, indent=2)}
        
        Please provide:
        1. Specific action items to improve each KPI
        2. Priority level for each recommendation
        3. Expected impact
        4. Implementation timeline
        5. Required resources
        
        Focus on practical, actionable recommendations.
        Respond in JSON format with structured recommendations.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business strategy expert. Provide practical recommendations for improving KPIs."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            self.recommendations = recommendations
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return {"error": "Recommendation generation failed"}
    
    def forecast_kpis(self, kpis: Dict[str, pd.Series], periods: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for KPIs using Prophet.
        
        Args:
            kpis (Dict[str, pd.Series]): Dictionary of calculated KPIs
            periods (int): Number of periods to forecast
            
        Returns:
            Dict[str, pd.DataFrame]: Forecasts for each KPI
        """
        forecasts = {}
        
        for kpi_name, kpi_values in kpis.items():
            try:
                # Prepare data for Prophet
                df_prophet = pd.DataFrame({
                    'ds': kpi_values.index,
                    'y': kpi_values.values
                })
                
                # Create and fit model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                model.fit(df_prophet)
                
                # Make forecast
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                forecasts[kpi_name] = forecast
                
            except Exception as e:
                logging.error(f"Error forecasting {kpi_name}: {str(e)}")
        
        self.forecasts = forecasts
        return forecasts
    
    def create_dashboard(self, kpis: Dict[str, pd.Series]) -> None:
        """
        Create an interactive dashboard using Streamlit.
        """
        st.title("AI-Powered KPI Dashboard")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", 
            ["Overview", "Detailed Analysis", "Forecasts", "Recommendations"])
        
        if page == "Overview":
            st.header("KPI Overview")
            
            # Create KPI cards
            cols = st.columns(len(kpis))
            for i, (kpi_name, kpi_values) in enumerate(kpis.items()):
                with cols[i]:
                    current_value = kpi_values.iloc[-1]
                    previous_value = kpi_values.iloc[-2]
                    change_pct = ((current_value - previous_value) / previous_value) * 100
                    
                    st.metric(
                        label=kpi_name.replace('_', ' ').title(),
                        value=f"{current_value:.2f}",
                        delta=f"{change_pct:.1f}%"
                    )
            
            # Show recent trends
            st.subheader("Recent Trends")
            for kpi_name, kpi_values in kpis.items():
                fig = px.line(
                    x=kpi_values.index[-30:],
                    y=kpi_values.values[-30:],
                    title=f"{kpi_name.replace('_', ' ').title()} - Last 30 Days"
                )
                st.plotly_chart(fig)
        
        elif page == "Detailed Analysis":
            st.header("Detailed KPI Analysis")
            
            # Show trend analysis
            if self.insights:
                for category, details in self.insights[-1][1].items():
                    st.subheader(category.replace('_', ' ').title())
                    if isinstance(details, list):
                        for item in details:
                            st.write(f"- {item}")
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            st.write(f"- {key}: {value}")
                    else:
                        st.write(details)
            
            # Show correlation matrix
            st.subheader("KPI Correlations")
            kpi_df = pd.DataFrame(kpis)
            corr_matrix = kpi_df.corr()
            fig = px.imshow(
                corr_matrix,
                title="KPI Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig)
        
        elif page == "Forecasts":
            st.header("KPI Forecasts")
            
            # Show forecasts
            for kpi_name, forecast in self.forecasts.items():
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=kpis[kpi_name].index,
                    y=kpis[kpi_name].values,
                    name="Actual",
                    mode="lines"
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name="Forecast",
                    mode="lines",
                    line=dict(dash='dash')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{kpi_name.replace('_', ' ').title()} - Forecast",
                    xaxis_title="Date",
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig)
        
        elif page == "Recommendations":
            st.header("AI-Powered Recommendations")
            
            if self.recommendations:
                for category, items in self.recommendations.items():
                    st.subheader(category.replace('_', ' ').title())
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                st.write(f"**{item.get('title', 'Recommendation')}**")
                                st.write(f"Priority: {item.get('priority', 'N/A')}")
                                st.write(f"Impact: {item.get('impact', 'N/A')}")
                                st.write(f"Timeline: {item.get('timeline', 'N/A')}")
                                st.write(f"Details: {item.get('details', '')}")
                                st.write("---")
                            else:
                                st.write(f"- {item}")
                    else:
                        st.write(items)
    
    def run_analysis(self) -> None:
        """
        Run the complete KPI analysis pipeline.
        """
        # 1. Calculate KPIs
        kpis = self.calculate_kpis()
        
        # 2. Analyze trends
        analysis = self.analyze_trends(kpis)
        
        # 3. Generate recommendations
        self.generate_recommendations(analysis)
        
        # 4. Generate forecasts
        self.forecast_kpis(kpis)
        
        # 5. Create dashboard
        self.create_dashboard(kpis)

def main():
    """
    Example usage of the KPITracker class.
    """
    try:
        # Load sample dataset
        df = pd.read_csv('data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Initialize and run KPI tracker
        tracker = KPITracker(df)
        tracker.run_analysis()
        
    except FileNotFoundError:
        st.error("Error: data.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 