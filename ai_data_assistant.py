"""
AI-Powered Data Assistant Tool
===========================

This script provides an intelligent assistant for data analysis tasks.
It helps automate common data analysis workflows and provides insights.

Author: Sujal Dhungana , Manish Paneru
License: MIT
"""

import streamlit as st
from openai import OpenAI
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
import sqlparse
import json
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataAnalystAssistant:
    """
    A specialized chatbot that helps data analysts with various tasks.
    """
    
    def __init__(self):
        """
        Initialize the Data Analyst Assistant.
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory()
        
        # Load knowledge base
        self.knowledge_base = self.load_knowledge_base()
        
        # Create output directory
        self.output_path = Path('assistant_output')
        self.output_path.mkdir(exist_ok=True)
    
    def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load the knowledge base for common data analysis concepts and solutions.
        
        Returns:
            Dict[str, Any]: Knowledge base content
        """
        return {
            'python': {
                'pandas': {
                    'common_operations': [
                        'df.head()', 'df.describe()', 'df.groupby()',
                        'df.merge()', 'df.pivot_table()', 'df.melt()'
                    ],
                    'examples': {
                        'groupby': "df.groupby('category')['value'].mean()",
                        'pivot': "df.pivot_table(index='date', columns='category', values='amount')",
                        'merge': "pd.merge(df1, df2, on='key', how='left')"
                    }
                },
                'numpy': {
                    'common_operations': [
                        'np.mean()', 'np.std()', 'np.array()',
                        'np.concatenate()', 'np.reshape()', 'np.random.rand()'
                    ],
                    'examples': {
                        'array_ops': "np.array([1, 2, 3]).mean()",
                        'matrix_ops': "np.dot(matrix1, matrix2)",
                        'random': "np.random.normal(0, 1, 1000)"
                    }
                }
            },
            'sql': {
                'common_queries': {
                    'select': "SELECT column FROM table WHERE condition",
                    'join': "SELECT t1.col, t2.col FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id",
                    'group_by': "SELECT category, COUNT(*) FROM table GROUP BY category",
                    'window': "SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY value) FROM table"
                },
                'best_practices': [
                    "Use meaningful table aliases",
                    "Write readable, formatted queries",
                    "Use appropriate indexing",
                    "Optimize JOIN operations"
                ]
            },
            'statistics': {
                'concepts': {
                    'hypothesis_testing': {
                        'description': "Process of testing a claim about a population parameter",
                        'examples': ["t-test", "chi-square test", "ANOVA"]
                    },
                    'regression': {
                        'description': "Statistical method for modeling relationships between variables",
                        'types': ["Linear", "Logistic", "Multiple"]
                    }
                },
                'formulas': {
                    'mean': "sum(x) / n",
                    'std_dev': "sqrt(sum((x - mean)^2) / (n-1))",
                    'correlation': "cov(x,y) / (std_dev(x) * std_dev(y))"
                }
            }
        }
    
    def format_code(self, code: str, language: str) -> str:
        """
        Format code snippets for better readability.
        
        Args:
            code (str): Code to format
            language (str): Programming language
            
        Returns:
            str: Formatted code
        """
        if language.lower() == 'sql':
            return sqlparse.format(
                code,
                reindent=True,
                keyword_case='upper'
            )
        elif language.lower() == 'python':
            # Basic Python formatting
            lines = code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                # Adjust indent level based on content
                if re.search(r':\s*$', line):
                    formatted_lines.append('    ' * indent_level + line.strip())
                    indent_level += 1
                elif line.strip().startswith(('return', 'break', 'continue')):
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append('    ' * indent_level + line.strip())
                else:
                    formatted_lines.append('    ' * indent_level + line.strip())
            
            return '\n'.join(formatted_lines)
        
        return code
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a data analysis related query.
        
        Args:
            query (str): User's question or request
            
        Returns:
            Dict[str, Any]: Response with explanation and code examples
        """
        # Prepare context from knowledge base
        context = {
            'query': query,
            'knowledge_base': self.knowledge_base,
            'conversation_history': self.memory.buffer
        }
        
        prompt = f"""
        As a data analysis expert, help with this query:
        {json.dumps(context, indent=2)}
        
        Please provide:
        1. Clear explanation
        2. Code examples (if applicable)
        3. Best practices
        4. Common pitfalls to avoid
        5. Additional resources
        
        If the query is about code, include working examples.
        If it's about statistics, include intuitive explanations.
        If it's about best practices, provide concrete examples.
        
        Respond in JSON format with structured information.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst assistant. Provide clear, practical help."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse and format response
            content = json.loads(response.choices[0].message.content)
            
            # Format any code snippets in the response
            if 'code_examples' in content:
                for lang, code in content['code_examples'].items():
                    content['code_examples'][lang] = self.format_code(code, lang)
            
            # Update conversation memory
            self.memory.save_context(
                {"input": query},
                {"output": json.dumps(content)}
            )
            
            return content
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {
                "error": "Failed to generate response",
                "details": str(e)
            }
    
    def create_interface(self) -> None:
        """
        Create an interactive Streamlit interface for the assistant.
        """
        st.title("AI Data Analyst Assistant")
        st.write("""
        Welcome! I'm your AI assistant for data analysis tasks.
        I can help you with:
        - Python coding
        - SQL queries
        - Statistical concepts
        - Data analysis workflows
        - Best practices
        """)
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    content = message["content"]
                    if isinstance(content, dict):
                        if "explanation" in content:
                            st.write("📝 Explanation:")
                            st.write(content["explanation"])
                        
                        if "code_examples" in content:
                            st.write("💻 Code Examples:")
                            for lang, code in content["code_examples"].items():
                                st.code(code, language=lang.lower())
                        
                        if "best_practices" in content:
                            st.write("✨ Best Practices:")
                            for practice in content["best_practices"]:
                                st.write(f"- {practice}")
                        
                        if "resources" in content:
                            st.write("📚 Additional Resources:")
                            for resource in content["resources"]:
                                st.write(f"- {resource}")
                    else:
                        st.write(content)
                else:
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about data analysis!"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                response = self.generate_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                if "error" not in response:
                    if "explanation" in response:
                        st.write("📝 Explanation:")
                        st.write(response["explanation"])
                    
                    if "code_examples" in response:
                        st.write("💻 Code Examples:")
                        for lang, code in response["code_examples"].items():
                            st.code(code, language=lang.lower())
                    
                    if "best_practices" in response:
                        st.write("✨ Best Practices:")
                        for practice in response["best_practices"]:
                            st.write(f"- {practice}")
                    
                    if "resources" in response:
                        st.write("📚 Additional Resources:")
                        for resource in response["resources"]:
                            st.write(f"- {resource}")
                else:
                    st.error(f"Error: {response['error']}")
    
    def save_conversation(self) -> None:
        """
        Save the current conversation to a file.
        """
        if st.session_state.messages:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_path / f"conversation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(st.session_state.messages, f, indent=2)
            
            st.success(f"Conversation saved to {filename}")

def main():
    """
    Main function to run the Data Analyst Assistant.
    """
    try:
        assistant = DataAnalystAssistant()
        assistant.create_interface()
        
        # Add save button
        if st.sidebar.button("Save Conversation"):
            assistant.save_conversation()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 