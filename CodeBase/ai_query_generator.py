"""
AI-Powered SQL Query Generator Tutorial
====================================

This script demonstrates how to convert natural language questions into SQL queries
using OpenAI's GPT models. It helps analysts generate SQL queries without needing
to remember exact syntax.

Features:
1. Natural language to SQL conversion
2. Query validation and safety checks
3. Query explanation
4. Schema inference
5. Query optimization suggestions
6. Support for complex joins and aggregations

Author: [Your Name]
Date: [Current Date]
License: MIT
"""

import pandas as pd
import sqlite3
from openai import OpenAI
import json
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import sqlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
OPENAI_API_KEY = "your_api_key_here"
client = OpenAI(api_key=OPENAI_API_KEY)

class AIQueryGenerator:
    """
    A class that converts natural language questions into SQL queries using AI.
    It also provides query validation and explanation features.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the Query Generator.
        
        Args:
            db_path (str, optional): Path to SQLite database. If None, will create in-memory DB.
        """
        self.conn = sqlite3.connect(db_path if db_path else ':memory:')
        self.cursor = self.conn.cursor()
        self.tables_info = {}
        self.query_history = []
    
    def load_csv_to_table(self, csv_path: str, table_name: str) -> None:
        """
        Load a CSV file into a SQLite table.
        
        Args:
            csv_path (str): Path to CSV file
            table_name (str): Name for the table
        """
        try:
            # Read CSV and create table
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            
            # Store table information
            self.tables_info[table_name] = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(3).to_dict(orient='records')
            }
            
            logging.info(f"Successfully loaded {csv_path} into table {table_name}")
            
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            raise
    
    def get_schema_info(self) -> str:
        """
        Get formatted schema information for all tables.
        
        Returns:
            str: Formatted schema information
        """
        schema_info = []
        for table_name, info in self.tables_info.items():
            columns = [f"{col} ({dtype})" for col, dtype in info['dtypes'].items()]
            schema_info.append(f"Table: {table_name}")
            schema_info.append("Columns:")
            schema_info.extend([f"  - {col}" for col in columns])
            schema_info.append("Sample Data:")
            for row in info['sample_data']:
                schema_info.append(f"  {row}")
            schema_info.append("")
        
        return "\n".join(schema_info)
    
    def generate_sql_query(self, question: str) -> Dict[str, Any]:
        """
        Convert a natural language question into a SQL query using AI.
        
        Args:
            question (str): Natural language question
            
        Returns:
            Dict: Contains generated query and explanation
        """
        # Prepare context for the AI
        schema_info = self.get_schema_info()
        
        prompt = f"""
        Given this database schema and sample data:
        {schema_info}
        
        Convert this question into a SQL query:
        "{question}"
        
        Provide:
        1. SQL query
        2. Explanation of the query
        3. Any assumptions made
        4. Potential optimizations
        
        Respond in JSON format with these keys:
        - query: The SQL query
        - explanation: Step-by-step explanation
        - assumptions: List of assumptions
        - optimizations: List of potential optimizations
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in SQL query generation. Generate clear, efficient queries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Format the SQL query
            if 'query' in result:
                result['query'] = sqlparse.format(
                    result['query'],
                    reindent=True,
                    keyword_case='upper'
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Error generating SQL query: {str(e)}")
            return {
                "error": "Failed to generate query",
                "details": str(e)
            }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a SQL query using AI and basic checks.
        
        Args:
            query (str): SQL query to validate
            
        Returns:
            Dict: Validation results and suggestions
        """
        # Basic SQL injection prevention
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT']
        if any(keyword in query.upper() for keyword in dangerous_keywords):
            return {
                "is_safe": False,
                "warnings": ["Query contains potentially dangerous operations"]
            }
        
        prompt = f"""
        Validate this SQL query and check for potential issues:
        {query}
        
        Please check for:
        1. Syntax correctness
        2. Potential performance issues
        3. Best practices violations
        4. Missing indexes or constraints
        5. Security concerns
        
        Respond in JSON format with validation results.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a SQL query validator. Check for issues and suggest improvements."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            validation_results = json.loads(response.choices[0].message.content)
            
            # Try executing the query to check for runtime errors
            try:
                self.cursor.execute("EXPLAIN QUERY PLAN " + query)
                validation_results["execution_test"] = "Query plan generated successfully"
            except sqlite3.Error as e:
                validation_results["execution_test"] = f"Query execution error: {str(e)}"
            
            return validation_results
            
        except Exception as e:
            logging.error(f"Error validating query: {str(e)}")
            return {
                "error": "Validation failed",
                "details": str(e)
            }
    
    def execute_query(self, query: str) -> Tuple[List[str], List[Tuple]]:
        """
        Safely execute a SQL query and return results.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Tuple[List[str], List[Tuple]]: Column names and results
        """
        try:
            # Execute query
            self.cursor.execute(query)
            
            # Get column names and results
            columns = [description[0] for description in self.cursor.description]
            results = self.cursor.fetchall()
            
            # Store in query history
            self.query_history.append({
                'query': query,
                'timestamp': pd.Timestamp.now(),
                'num_results': len(results)
            })
            
            return columns, results
            
        except sqlite3.Error as e:
            logging.error(f"Error executing query: {str(e)}")
            raise
    
    def get_query_explanation(self, query: str) -> Dict[str, Any]:
        """
        Get a detailed explanation of what a SQL query does.
        
        Args:
            query (str): SQL query to explain
            
        Returns:
            Dict: Detailed explanation of the query
        """
        prompt = f"""
        Explain this SQL query in detail:
        {query}
        
        Please provide:
        1. Step-by-step explanation of what the query does
        2. Description of each clause and its purpose
        3. Expected output format
        4. Any potential performance implications
        
        Make the explanation suitable for a beginner SQL user.
        Respond in JSON format with structured explanation.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a SQL teacher. Explain queries in a clear, beginner-friendly way."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logging.error(f"Error getting query explanation: {str(e)}")
            return {
                "error": "Explanation failed",
                "details": str(e)
            }

def main():
    """
    Example usage of the AIQueryGenerator class.
    """
    try:
        # Initialize query generator
        generator = AIQueryGenerator()
        
        # Load sample dataset
        generator.load_csv_to_table('data.csv', 'sales')
        
        # Example questions to convert to SQL
        questions = [
            "What are the top 5 selling products?",
            "Show me monthly sales trends for the last year",
            "Which customers have spent more than $1000 in total?",
            "What is the average order value by customer segment?"
        ]
        
        print("AI-Powered SQL Query Generator Demo")
        print("="*50)
        
        for question in questions:
            print(f"\nQuestion: {question}")
            print("-" * 30)
            
            # Generate SQL query
            result = generator.generate_sql_query(question)
            
            if "error" not in result:
                print("\nGenerated SQL:")
                print(result['query'])
                print("\nExplanation:")
                print(result['explanation'])
                
                # Validate query
                validation = generator.validate_query(result['query'])
                if validation.get("is_safe", False):
                    try:
                        # Execute query
                        columns, results = generator.execute_query(result['query'])
                        print("\nResults:")
                        print(pd.DataFrame(results, columns=columns).head())
                    except sqlite3.Error as e:
                        print(f"\nError executing query: {str(e)}")
                else:
                    print("\nQuery validation failed:")
                    print(validation.get("warnings", ["Unknown validation error"]))
            else:
                print(f"\nError: {result['error']}")
            
            print("\n" + "="*50)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 