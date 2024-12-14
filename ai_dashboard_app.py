"""
AI-Powered Real-Time Dashboard Tutorial
=====================================

This script demonstrates how to create an interactive dashboard using Streamlit
where users can upload datasets and get AI-powered insights and visualizations.

The dashboard includes:
1. Data upload and preview
2. Automated data profiling
3. AI-generated insights
4. Dynamic visualizations
5. Pattern detection
6. Correlation analysis

Author: [Your Name]
Date: [Current Date]
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
from typing import Dict, List, Any
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_ai_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate AI-powered insights about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict: Dictionary containing various insights
    """
    # Prepare dataset statistics
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'summary_stats': df.describe().to_dict()
    }
    
    # Get AI insights
    prompt = f"""
    Analyze this dataset and provide key insights:
    {json.dumps(stats, indent=2)}
    
    Please provide:
    1. Main patterns and trends
    2. Notable relationships between variables
    3. Potential areas for further investigation
    4. Suggested visualizations
    5. Data quality observations
    
    Respond in JSON format with structured insights.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Provide clear, actionable insights."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return {"error": "Failed to generate insights"}

def create_visualization(df: pd.DataFrame, viz_type: str, columns: List[str]) -> go.Figure:
    """
    Create a Plotly visualization based on user selection.
    
    Args:
        df (pd.DataFrame): Input dataset
        viz_type (str): Type of visualization
        columns (List[str]): Columns to visualize
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        if viz_type == "Scatter Plot":
            fig = px.scatter(df, x=columns[0], y=columns[1],
                           title=f"{columns[0]} vs {columns[1]}")
        
        elif viz_type == "Line Plot":
            fig = px.line(df, x=columns[0], y=columns[1],
                         title=f"{columns[1]} Over {columns[0]}")
        
        elif viz_type == "Bar Plot":
            fig = px.bar(df, x=columns[0], y=columns[1],
                        title=f"{columns[1]} by {columns[0]}")
        
        elif viz_type == "Box Plot":
            fig = px.box(df, x=columns[0], y=columns[1],
                        title=f"Distribution of {columns[1]} by {columns[0]}")
        
        elif viz_type == "Histogram":
            fig = px.histogram(df, x=columns[0],
                             title=f"Distribution of {columns[0]}")
        
        elif viz_type == "Correlation Heatmap":
            corr_matrix = df[columns].corr()
            fig = px.imshow(corr_matrix, title="Correlation Matrix")
        
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def get_ai_visualization_suggestions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get AI suggestions for effective visualizations.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict: Visualization suggestions
    """
    # Analyze dataset characteristics
    data_info = {
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns},
        'has_temporal': any(col.lower().contains('date', 'time', 'year', 'month') 
                          for col in df.columns)
    }
    
    prompt = f"""
    Suggest effective visualizations for this dataset:
    {json.dumps(data_info, indent=2)}
    
    Please provide:
    1. Recommended visualization types
    2. Which columns to use for each visualization
    3. Why each visualization would be insightful
    4. Any specific settings or configurations
    
    Respond in JSON format with structured recommendations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Suggest effective ways to visualize this data."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        st.error(f"Error getting visualization suggestions: {str(e)}")
        return {"error": "Failed to get visualization suggestions"}

def main():
    """
    Main function to run the Streamlit dashboard.
    """
    st.title("AI-Powered Data Analysis Dashboard")
    st.write("""
    Upload your dataset and get instant AI-powered insights and visualizations.
    The dashboard will automatically analyze your data and suggest interesting patterns.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and cache data
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)
        
        df = load_data(uploaded_file)
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", 
            ["Data Preview", "AI Insights", "Interactive Visualizations", "Data Profiling"])
        
        if page == "Data Preview":
            st.header("Data Preview")
            st.write("First few rows of your dataset:")
            st.dataframe(df.head())
            
            st.write("Dataset Info:")
            st.write(f"- Rows: {df.shape[0]}")
            st.write(f"- Columns: {df.shape[1]}")
            st.write("- Column Types:")
            for col, dtype in df.dtypes.items():
                st.write(f"  - {col}: {dtype}")
        
        elif page == "AI Insights":
            st.header("AI-Generated Insights")
            
            with st.spinner("Generating insights..."):
                insights = generate_ai_insights(df)
            
            if "error" not in insights:
                # Display insights in an organized way
                for category, details in insights.items():
                    st.subheader(category.title())
                    if isinstance(details, list):
                        for item in details:
                            st.write(f"- {item}")
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            st.write(f"- {key}: {value}")
                    else:
                        st.write(details)
            else:
                st.error("Failed to generate insights. Please try again.")
        
        elif page == "Interactive Visualizations":
            st.header("Interactive Visualizations")
            
            # Get AI suggestions for visualizations
            with st.spinner("Getting visualization suggestions..."):
                viz_suggestions = get_ai_visualization_suggestions(df)
            
            # User inputs for visualization
            viz_type = st.selectbox("Select Visualization Type",
                ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", 
                 "Histogram", "Correlation Heatmap"])
            
            # Column selection based on visualization type
            if viz_type == "Histogram":
                cols = st.multiselect("Select Column", df.columns, max_selections=1)
            elif viz_type == "Correlation Heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                cols = st.multiselect("Select Columns", numeric_cols)
            else:
                cols = st.multiselect("Select Columns (X, Y)", df.columns, max_selections=2)
            
            # Create and display visualization
            if len(cols) > 0:
                fig = create_visualization(df, viz_type, cols)
                if fig is not None:
                    st.plotly_chart(fig)
            
            # Display AI suggestions
            if "error" not in viz_suggestions:
                st.subheader("AI Visualization Suggestions")
                for suggestion in viz_suggestions.get("suggestions", []):
                    st.write(f"- {suggestion}")
        
        elif page == "Data Profiling":
            st.header("Automated Data Profiling")
            
            with st.spinner("Generating comprehensive data profile..."):
                profile = df.profile_report()
                st_profile_report(profile)

if __name__ == "__main__":
    main() 