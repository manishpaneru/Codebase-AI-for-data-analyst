"""
AI-Powered Feature Engineering Tool
================================

This script demonstrates advanced feature engineering techniques using AI models.
It helps create and select optimal features for machine learning models.

Author: Sujal Dhungana , Manish Paneru
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from openai import OpenAI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import json
from category_encoders import TargetEncoder, WOEEncoder
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """
    A class that uses AI to assist in feature engineering tasks.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        """
        Initialize the Feature Engineer.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of target variable column
        """
        self.df = df.copy()
        self.target_column = target_column
        self.features = {}
        self.feature_importance = {}
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Create output directory
        self.output_path = Path('feature_output')
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize encoders
        self.encoders = {}
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze the dataset to understand its characteristics.
        
        Returns:
            Dict[str, Any]: Dataset analysis results
        """
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns}
        }
        
        if self.target_column:
            analysis['target_distribution'] = self.df[self.target_column].value_counts().to_dict()
        
        return analysis
    
    def suggest_features(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use AI to suggest potential features based on dataset analysis.
        
        Args:
            analysis (Dict[str, Any]): Dataset analysis results
            
        Returns:
            List[Dict[str, Any]]: Suggested features
        """
        prompt = f"""
        Suggest features to engineer based on this dataset analysis:
        {json.dumps(analysis, indent=2)}
        
        Consider:
        1. Numeric transformations
        2. Categorical encoding strategies
        3. Feature interactions
        4. Time-based features (if applicable)
        5. Domain-specific features
        
        Provide specific suggestions with implementation details.
        Respond in JSON format with structured feature suggestions.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a feature engineering expert. Suggest effective features for machine learning."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            suggestions = json.loads(response.choices[0].message.content)
            return suggestions.get('features', [])
            
        except Exception as e:
            logging.error(f"Error getting feature suggestions: {str(e)}")
            return []
    
    def create_numeric_features(self, column: str) -> pd.DataFrame:
        """
        Create features from numeric columns.
        
        Args:
            column (str): Column name
            
        Returns:
            pd.DataFrame: DataFrame with new features
        """
        features = pd.DataFrame()
        series = self.df[column]
        
        # Basic transformations
        features[f"{column}_log"] = np.log1p(series.clip(lower=0))
        features[f"{column}_sqrt"] = np.sqrt(series.clip(lower=0))
        features[f"{column}_square"] = np.square(series)
        
        # Binning
        features[f"{column}_bins"] = pd.qcut(series, q=10, labels=False, duplicates='drop')
        
        # Rolling statistics (if sorted data)
        if 'date' in self.df.columns:
            self.df = self.df.sort_values('date')
            features[f"{column}_rolling_mean"] = series.rolling(window=7).mean()
            features[f"{column}_rolling_std"] = series.rolling(window=7).std()
        
        return features
    
    def create_categorical_features(self, column: str) -> pd.DataFrame:
        """
        Create features from categorical columns.
        
        Args:
            column (str): Column name
            
        Returns:
            pd.DataFrame: DataFrame with new features
        """
        features = pd.DataFrame()
        series = self.df[column]
        
        # Count encoding
        count_map = series.value_counts()
        features[f"{column}_count"] = series.map(count_map)
        
        # Frequency encoding
        freq_map = series.value_counts(normalize=True)
        features[f"{column}_freq"] = series.map(freq_map)
        
        # Target encoding (if target column exists)
        if self.target_column:
            encoder = TargetEncoder()
            features[f"{column}_target"] = encoder.fit_transform(series, self.df[self.target_column])
            self.encoders[f"{column}_target"] = encoder
        
        # One-hot encoding for low cardinality
        if series.nunique() < 10:
            onehot = pd.get_dummies(series, prefix=column)
            features = pd.concat([features, onehot], axis=1)
        
        return features
    
    def create_interaction_features(self, columns: List[str]) -> pd.DataFrame:
        """
        Create interaction features between columns.
        
        Args:
            columns (List[str]): List of column names
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        features = pd.DataFrame()
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                
                # Numeric interactions
                if (self.df[col1].dtype in ['int64', 'float64'] and 
                    self.df[col2].dtype in ['int64', 'float64']):
                    features[f"{col1}_{col2}_sum"] = self.df[col1] + self.df[col2]
                    features[f"{col1}_{col2}_diff"] = self.df[col1] - self.df[col2]
                    features[f"{col1}_{col2}_product"] = self.df[col1] * self.df[col2]
                    features[f"{col1}_{col2}_ratio"] = self.df[col1] / (self.df[col2] + 1e-6)
                
                # Categorical interactions
                elif (self.df[col1].dtype == 'object' and 
                      self.df[col2].dtype == 'object'):
                    features[f"{col1}_{col2}_combined"] = self.df[col1] + "_" + self.df[col2]
        
        return features
    
    def select_features(self, features: pd.DataFrame, n_features: int = 10) -> List[str]:
        """
        Select most important features using multiple methods.
        
        Args:
            features (pd.DataFrame): Feature DataFrame
            n_features (int): Number of features to select
            
        Returns:
            List[str]: Selected feature names
        """
        if not self.target_column:
            return list(features.columns[:n_features])
        
        selected_features = set()
        
        try:
            # 1. Mutual Information
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            mi_selector.fit(features, self.df[self.target_column])
            mi_selected = features.columns[mi_selector.get_support()].tolist()
            selected_features.update(mi_selected)
            
            # 2. Random Forest Importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, self.df[self.target_column])
            rf_importance = pd.Series(rf.feature_importances_, index=features.columns)
            rf_selected = rf_importance.nlargest(n_features).index.tolist()
            selected_features.update(rf_selected)
            
            # 3. Recursive Feature Elimination
            rfe = RFE(estimator=rf, n_features_to_select=n_features)
            rfe.fit(features, self.df[self.target_column])
            rfe_selected = features.columns[rfe.support_].tolist()
            selected_features.update(rfe_selected)
            
            # Store feature importance
            self.feature_importance = {
                'mutual_info': dict(zip(mi_selected, mi_selector.scores_)),
                'random_forest': dict(zip(features.columns, rf.feature_importances_)),
                'rfe': dict(zip(features.columns, rfe.ranking_))
            }
            
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
        
        return list(selected_features)[:n_features]
    
    def optimize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize feature engineering parameters using Optuna.
        
        Args:
            features (pd.DataFrame): Feature DataFrame
            
        Returns:
            pd.DataFrame: Optimized features
        """
        if not self.target_column:
            return features
        
        def objective(trial):
            # Example parameters to optimize
            n_components = trial.suggest_int('n_components', 2, min(50, features.shape[1]))
            
            # PCA transformation
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(StandardScaler().fit_transform(features))
            
            # Evaluate using Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            score = np.mean(cross_val_score(rf, transformed, self.df[self.target_column], cv=5))
            
            return score
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            
            # Apply best parameters
            best_n_components = study.best_params['n_components']
            pca = PCA(n_components=best_n_components)
            optimized = pd.DataFrame(
                pca.fit_transform(StandardScaler().fit_transform(features)),
                columns=[f'pca_{i}' for i in range(best_n_components)]
            )
            
            return optimized
            
        except Exception as e:
            logging.error(f"Error in feature optimization: {str(e)}")
            return features
    
    def create_features(self) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Returns:
            pd.DataFrame: Engineered features
        """
        try:
            # 1. Analyze dataset
            analysis = self.analyze_dataset()
            
            # 2. Get AI suggestions
            suggestions = self.suggest_features(analysis)
            
            # 3. Create features
            all_features = pd.DataFrame()
            
            # Numeric features
            for column in analysis['numeric_columns']:
                features = self.create_numeric_features(column)
                all_features = pd.concat([all_features, features], axis=1)
            
            # Categorical features
            for column in analysis['categorical_columns']:
                features = self.create_categorical_features(column)
                all_features = pd.concat([all_features, features], axis=1)
            
            # Interaction features
            interaction_features = self.create_interaction_features(
                analysis['numeric_columns'] + analysis['categorical_columns']
            )
            all_features = pd.concat([all_features, interaction_features], axis=1)
            
            # 4. Select features
            selected_features = self.select_features(all_features)
            selected_df = all_features[selected_features]
            
            # 5. Optimize features
            optimized_df = self.optimize_features(selected_df)
            
            return optimized_df
            
        except Exception as e:
            logging.error(f"Error in feature engineering pipeline: {str(e)}")
            return pd.DataFrame()
    
    def create_feature_report(self, features: pd.DataFrame) -> str:
        """
        Generate a comprehensive feature engineering report.
        
        Args:
            features (pd.DataFrame): Engineered features
            
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'original_shape': self.df.shape,
            'engineered_shape': features.shape,
            'feature_importance': self.feature_importance,
            'features_created': list(features.columns),
            'feature_stats': features.describe().to_dict()
        }
        
        prompt = f"""
        Generate a feature engineering report based on these results:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Overview of feature engineering process
        2. Summary of created features
        3. Feature importance analysis
        4. Statistical properties
        5. Recommendations for further improvements
        
        Format the report in Markdown, including sections for visualizations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a feature engineering expert. Create clear, informative reports."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'feature_engineering_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def create_visualizations(self, features: pd.DataFrame) -> None:
        """
        Create visualizations of the engineered features.
        """
        st.title("Feature Engineering Results")
        
        # 1. Feature Importance
        st.header("Feature Importance")
        if self.feature_importance:
            # Random Forest importance
            rf_importance = pd.DataFrame({
                'feature': list(self.feature_importance['random_forest'].keys()),
                'importance': list(self.feature_importance['random_forest'].values())
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                rf_importance,
                x='feature',
                y='importance',
                title="Random Forest Feature Importance"
            )
            st.plotly_chart(fig)
        
        # 2. Feature Correlations
        st.header("Feature Correlations")
        corr_matrix = features.corr()
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig)
        
        # 3. Feature Distributions
        st.header("Feature Distributions")
        for column in features.select_dtypes(include=[np.number]).columns:
            fig = px.histogram(
                features,
                x=column,
                title=f"Distribution of {column}"
            )
            st.plotly_chart(fig)
        
        # 4. Target Relationship (if exists)
        if self.target_column:
            st.header("Relationship with Target")
            for column in features.select_dtypes(include=[np.number]).columns:
                fig = px.scatter(
                    x=features[column],
                    y=self.df[self.target_column],
                    title=f"{column} vs {self.target_column}"
                )
                st.plotly_chart(fig)

def main():
    """
    Main function to run the Feature Engineer.
    """
    try:
        # Load sample dataset
        df = pd.read_csv('data.csv')
        
        st.title("AI-Powered Feature Engineering")
        st.write("""
        This tool helps you create and select optimal features for your machine learning models.
        Upload your dataset and let AI suggest and engineer features for you.
        """)
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Select target column
            target_column = st.selectbox(
                "Select target column (optional)",
                [None] + list(df.columns)
            )
            
            # Initialize feature engineer
            engineer = FeatureEngineer(df, target_column)
            
            if st.button("Engineer Features"):
                with st.spinner("Engineering features..."):
                    # Create features
                    features = engineer.create_features()
                    
                    # Generate report
                    report = engineer.create_feature_report(features)
                    
                    # Show results
                    st.success(f"Created {features.shape[1]} features!")
                    
                    # Display report
                    st.markdown(report)
                    
                    # Show visualizations
                    engineer.create_visualizations(features)
                    
                    # Download features
                    st.download_button(
                        "Download Engineered Features",
                        features.to_csv(index=False),
                        "engineered_features.csv",
                        "text/csv"
                    )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 