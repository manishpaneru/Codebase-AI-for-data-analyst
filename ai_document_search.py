"""
Document Search for Data Analysts Tutorial
======================================

This script demonstrates how to build an intelligent document search system
that helps data analysts find relevant documentation, guides, and research
using semantic search and AI-powered insights.

Features:
1. Semantic document search
2. Content summarization
3. Relevance ranking
4. Document categorization
5. Interactive interface
6. Search history tracking
7. Export capabilities

Author: [Your Name]
Date: [Current Date]
License: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI
import streamlit as st
import plotly.express as px
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from typing import Dict, List, Tuple, Any
import logging
import shutil
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentSearch:
    """
    A class that provides intelligent document search capabilities for data analysts.
    """
    
    def __init__(self, docs_path: str):
        """
        Initialize the Document Search system.
        
        Args:
            docs_path (str): Path to directory containing documents
        """
        self.docs_path = Path(docs_path)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chroma_db"
        ))
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=st.secrets["OPENAI_API_KEY"],
            model_name="text-embedding-ada-002"
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="data_docs",
            embedding_function=self.embedding_function
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create output directory
        self.output_path = Path('search_output')
        self.output_path.mkdir(exist_ok=True)
        
        # Load search history
        self.search_history = self.load_search_history()
    
    def load_search_history(self) -> List[Dict[str, Any]]:
        """
        Load search history from file.
        
        Returns:
            List[Dict[str, Any]]: Search history
        """
        history_file = self.output_path / 'search_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_search_history(self) -> None:
        """
        Save search history to file.
        """
        history_file = self.output_path / 'search_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.search_history, f, indent=2)
    
    def index_documents(self) -> None:
        """
        Index documents in the specified directory.
        """
        try:
            # Clear existing collection
            self.collection.delete(where={})
            
            # Process each document
            for file_path in self.docs_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.pdf', '.py']:
                    try:
                        # Read document content
                        content = file_path.read_text(encoding='utf-8')
                        
                        # Split into chunks
                        chunks = self.text_splitter.split_text(content)
                        
                        # Add chunks to collection
                        for i, chunk in enumerate(chunks):
                            self.collection.add(
                                documents=[chunk],
                                metadatas=[{
                                    'source': str(file_path),
                                    'chunk_id': i,
                                    'total_chunks': len(chunks)
                                }],
                                ids=[f"{file_path.stem}_{i}"]
                            )
                        
                        logging.info(f"Indexed {file_path}")
                        
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {str(e)}")
            
            self.chroma_client.persist()
            logging.info("Document indexing complete")
            
        except Exception as e:
            logging.error(f"Error during indexing: {str(e)}")
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using semantic search.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Process results
            processed_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                processed_results.append(result)
            
            # Add to search history
            self.search_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'num_results': len(processed_results)
            })
            self.save_search_history()
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []
    
    def generate_summary(self, content: str) -> str:
        """
        Generate a summary of document content using AI.
        
        Args:
            content (str): Content to summarize
            
        Returns:
            str: Generated summary
        """
        prompt = f"""
        Summarize this document content for a data analyst:
        {content[:2000]}  # Limit content length for API
        
        Please provide:
        1. Main topics covered
        2. Key points
        3. Relevant use cases
        4. Technical details
        
        Keep the summary concise and focused on data analysis aspects.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert. Create clear, focused summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return "Error generating summary"
    
    def categorize_document(self, content: str) -> List[str]:
        """
        Categorize document content using AI.
        
        Args:
            content (str): Content to categorize
            
        Returns:
            List[str]: Assigned categories
        """
        prompt = f"""
        Categorize this document content for a data analyst:
        {content[:2000]}  # Limit content length for API
        
        Assign relevant categories from:
        - Data Analysis
        - Machine Learning
        - Statistics
        - Data Visualization
        - Data Engineering
        - Best Practices
        - Tutorial
        - Reference
        
        Respond with a JSON array of applicable categories.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a documentation categorization expert. Assign precise, relevant categories."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logging.error(f"Error categorizing document: {str(e)}")
            return ["Uncategorized"]
    
    def create_interface(self) -> None:
        """
        Create an interactive search interface using Streamlit.
        """
        st.title("AI-Powered Document Search")
        st.write("""
        Search through documentation, guides, and research papers.
        Get AI-powered summaries and insights from your results.
        """)
        
        # Sidebar controls
        st.sidebar.title("Controls")
        
        if st.sidebar.button("Reindex Documents"):
            with st.spinner("Indexing documents..."):
                self.index_documents()
            st.success("Indexing complete!")
        
        n_results = st.sidebar.slider("Number of results", 1, 20, 5)
        
        # Main search interface
        query = st.text_input("Enter your search query:")
        
        if query:
            with st.spinner("Searching..."):
                results = self.search_documents(query, n_results)
            
            if results:
                st.subheader(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}: {Path(result['metadata']['source']).name}"):
                        # Show content preview
                        st.text_area("Content Preview", result['content'], height=100)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"Generate Summary {i}"):
                                summary = self.generate_summary(result['content'])
                                st.write("Summary:")
                                st.write(summary)
                        
                        with col2:
                            if st.button(f"Show Categories {i}"):
                                categories = self.categorize_document(result['content'])
                                st.write("Categories:")
                                for category in categories:
                                    st.write(f"- {category}")
                        
                        # Show metadata
                        st.write("Metadata:")
                        st.json(result['metadata'])
                        
                        # Show source link
                        source_path = result['metadata']['source']
                        st.write(f"Source: [{Path(source_path).name}]({source_path})")
            else:
                st.warning("No results found")
        
        # Search history visualization
        st.header("Search History")
        if self.search_history:
            history_df = pd.DataFrame(self.search_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Plot search activity
            fig = px.line(
                history_df,
                x='timestamp',
                y='num_results',
                title="Search Activity Over Time"
            )
            st.plotly_chart(fig)
            
            # Show recent searches
            st.subheader("Recent Searches")
            recent_searches = history_df.tail(10).sort_values('timestamp', ascending=False)
            for _, row in recent_searches.iterrows():
                st.write(f"- {row['query']} ({row['num_results']} results)")
    
    def export_results(self, results: List[Dict[str, Any]], format: str = 'json') -> str:
        """
        Export search results in specified format.
        
        Args:
            results (List[Dict[str, Any]]): Search results to export
            format (str): Export format ('json' or 'csv')
            
        Returns:
            str: Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            output_file = self.output_path / f"search_results_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format == 'csv':
            output_file = self.output_path / f"search_results_{timestamp}.csv"
            df = pd.DataFrame([
                {
                    'content': r['content'],
                    'source': r['metadata']['source'],
                    'chunk_id': r['metadata']['chunk_id'],
                    'distance': r['distance']
                }
                for r in results
            ])
            df.to_csv(output_file, index=False)
        
        return str(output_file)

def main():
    """
    Main function to run the Document Search system.
    """
    try:
        # Initialize search system
        docs_path = "docs"  # Update with your documents directory
        search = DocumentSearch(docs_path)
        
        # Create interface
        search.create_interface()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 