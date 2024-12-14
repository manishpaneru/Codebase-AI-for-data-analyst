"""
AI-Assisted Predictive Modeling Tutorial
=======================================

This script demonstrates how to use AI to assist in building and optimizing predictive models.
It covers a complete machine learning pipeline with AI assistance at each step:

1. Data Preprocessing and Feature Engineering
2. Model Selection
3. Hyperparameter Optimization
4. Model Evaluation and Interpretation

The script uses a loan default prediction dataset as an example, but the techniques
can be applied to any binary classification problem (e.g., churn prediction).

Author: [Your Name]
Date: [Current Date]
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import optuna  # For intelligent hyperparameter optimization
from openai import OpenAI
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, List, Tuple, Any
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging to track the model building process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client - Replace with your API key
OPENAI_API_KEY = "your_api_key_here"
client = OpenAI(api_key=OPENAI_API_KEY)

class AIModelAssistant:
    """
    A class that uses AI to assist in building and optimizing predictive models.
    It combines traditional machine learning with GPT-4 insights for better model development.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Initialize the AI Model Assistant.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target variable column
        """
        self.df = df.copy()
        self.target_column = target_column
        self.insights = []
        self.output_path = Path('model_output')
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize preprocessing objects
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Store best model and parameters
        self.best_model = None
        self.best_params = None
        self.feature_importance = None
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Use AI to analyze data quality and suggest preprocessing steps.
        This helps identify potential issues before model building.
        
        Returns:
            Dict: Data quality insights and recommendations
        """
        # Calculate basic statistics
        stats = {
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns},
            'target_distribution': self.df[self.target_column].value_counts().to_dict()
        }
        
        # Get AI insights about data quality
        prompt = f"""
        Analyze this dataset's quality metrics and provide recommendations:
        {json.dumps(stats, indent=2)}
        
        Please provide insights about:
        1. Data quality issues that need attention
        2. Recommended preprocessing steps
        3. Potential feature engineering ideas
        4. Class imbalance handling suggestions
        
        Focus on preparing the data for a {self.target_column} prediction model.
        Respond in JSON format with structured recommendations.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in machine learning and data preprocessing."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            preprocessing_insights = json.loads(response.choices[0].message.content)
            self.insights.append(('Data Quality Analysis', preprocessing_insights))
            
            return preprocessing_insights
            
        except Exception as e:
            logging.error(f"Error in data quality analysis: {str(e)}")
            return {"error": "Data quality analysis failed"}
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data based on AI recommendations.
        Handles missing values, encoding, and scaling.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed features and target variable
        """
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Handle missing values
        for column in X.columns:
            if X[column].isnull().any():
                if X[column].dtype in ['int64', 'float64']:
                    X[column].fillna(X[column].mean(), inplace=True)
                else:
                    X[column].fillna(X[column].mode()[0], inplace=True)
        
        # Encode categorical variables
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            self.label_encoders[column] = le
        
        # Scale numerical features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X.values, y.values
    
    def get_model_recommendations(self) -> Dict[str, Any]:
        """
        Use AI to recommend suitable models based on the dataset characteristics.
        
        Returns:
            Dict: Model recommendations and explanations
        """
        dataset_info = {
            'n_samples': len(self.df),
            'n_features': len(self.df.columns) - 1,
            'target_type': 'binary' if self.df[self.target_column].nunique() == 2 else 'multiclass',
            'data_characteristics': {
                'has_missing_values': self.df.isnull().any().any(),
                'has_categorical': (self.df.dtypes == 'object').any(),
                'target_balance': self.df[self.target_column].value_counts(normalize=True).to_dict()
            }
        }
        
        prompt = f"""
        Recommend machine learning models for this dataset:
        {json.dumps(dataset_info, indent=2)}
        
        Please provide:
        1. Top 3 recommended models with explanations
        2. Pros and cons of each model
        3. Suggested evaluation metrics
        4. Potential challenges to watch for
        
        Respond in JSON format with structured recommendations.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in machine learning model selection."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            model_recommendations = json.loads(response.choices[0].message.content)
            self.insights.append(('Model Recommendations', model_recommendations))
            
            return model_recommendations
            
        except Exception as e:
            logging.error(f"Error in getting model recommendations: {str(e)}")
            return {"error": "Model recommendation failed"}
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Use Optuna for intelligent hyperparameter optimization.
        This method tries different hyperparameter combinations to find the best model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            
        Returns:
            Dict: Best hyperparameters and optimization results
        """
        def objective(trial):
            """
            Optuna objective function for hyperparameter optimization.
            """
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params, random_state=42)
            
            # Use cross-validation for more robust evaluation
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Get best parameters
        self.best_params = study.best_params
        
        # Train final model with best parameters
        self.best_model = xgb.XGBClassifier(**self.best_params, random_state=42)
        self.best_model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = dict(zip(
            self.df.drop(columns=[self.target_column]).columns,
            self.best_model.feature_importances_
        ))
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'feature_importance': self.feature_importance
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model and get AI-powered interpretation of results.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            Dict: Evaluation metrics and AI interpretation
        """
        # Get predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Get AI interpretation of results
        prompt = f"""
        Analyze these model evaluation results:
        {json.dumps(metrics, indent=2)}
        
        Also analyze these feature importances:
        {json.dumps(self.feature_importance, indent=2)}
        
        Please provide:
        1. Interpretation of model performance
        2. Key insights about important features
        3. Suggestions for model improvement
        4. Potential deployment considerations
        
        Respond in JSON format with structured insights.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in machine learning model evaluation."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            evaluation_insights = json.loads(response.choices[0].message.content)
            self.insights.append(('Model Evaluation', evaluation_insights))
            
            # Create evaluation visualizations
            self._create_evaluation_plots(metrics, y_test, y_pred, y_pred_proba)
            
            return {
                'metrics': metrics,
                'insights': evaluation_insights
            }
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            return {"error": "Model evaluation failed"}
    
    def _create_evaluation_plots(self, metrics: Dict, y_test: np.ndarray, 
                               y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Create and save evaluation plots.
        
        Args:
            metrics (Dict): Evaluation metrics
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Prediction probabilities
        """
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(self.output_path / 'confusion_matrix.png')
        plt.close()
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(self.output_path / 'roc_curve.png')
        plt.close()
        
        # 3. Feature Importance Plot
        importance_df = pd.DataFrame(
            self.feature_importance.items(),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(self.output_path / 'feature_importance.png')
        plt.close()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive model development report using AI.
        
        Returns:
            str: Markdown-formatted report
        """
        report_context = {
            'dataset_info': {
                'n_samples': len(self.df),
                'n_features': len(self.df.columns) - 1,
                'target': self.target_column
            },
            'insights': self.insights
        }
        
        prompt = f"""
        Generate a comprehensive machine learning model development report based on these insights:
        {json.dumps(report_context, indent=2)}
        
        The report should include:
        1. Executive Summary
        2. Data Preprocessing Steps
        3. Model Selection Rationale
        4. Hyperparameter Optimization Results
        5. Model Performance Analysis
        6. Feature Importance Analysis
        7. Recommendations for Improvement
        
        Format the report in Markdown, including sections for visualizations.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a machine learning expert writing a model development report."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.choices[0].message.content
            
            # Save report
            with open(self.output_path / 'model_development_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error in report generation: {str(e)}")
            return "Error generating report"
    
    def run_complete_pipeline(self) -> Tuple[str, Path]:
        """
        Run the complete model development pipeline with AI assistance.
        
        Returns:
            Tuple[str, Path]: Development report and path to output directory
        """
        logging.info("Starting AI-assisted model development pipeline...")
        
        # 1. Analyze data quality
        preprocessing_insights = self.analyze_data_quality()
        logging.info("Data quality analysis complete")
        
        # 2. Preprocess data
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data preprocessing complete")
        
        # 3. Get model recommendations
        model_recommendations = self.get_model_recommendations()
        logging.info("Model recommendations received")
        
        # 4. Optimize model
        optimization_results = self.optimize_hyperparameters(X_train, y_train)
        logging.info("Hyperparameter optimization complete")
        
        # 5. Evaluate model
        evaluation_results = self.evaluate_model(X_test, y_test)
        logging.info("Model evaluation complete")
        
        # 6. Generate report
        report = self.generate_report()
        logging.info(f"Pipeline complete. Output saved to {self.output_path}")
        
        return report, self.output_path

def main():
    """
    Example usage of the AIModelAssistant class for loan default prediction.
    This demonstrates how to use AI assistance in building a predictive model.
    """
    try:
        # Load sample loan default dataset
        # You can replace this with your own dataset
        df = pd.read_csv('loan_data.csv')
        
        print("Starting AI-Assisted Model Development...")
        assistant = AIModelAssistant(df, target_column='loan_default')
        report, output_path = assistant.run_complete_pipeline()
        
        print(f"\nModel development complete! Check the following locations for results:")
        print(f"- Report: {output_path / 'model_development_report.md'}")
        print(f"- Visualizations: {output_path}")
        print("\nReport Preview:")
        print("="*50)
        print(report[:500] + "...\n")
        
    except FileNotFoundError:
        print("Error: loan_data.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 