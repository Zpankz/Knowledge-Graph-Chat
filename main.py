"""
Main entry point for the Knowledge Graph application.
"""
import logging
import logging.config
import os
from pathlib import Path
from dotenv import load_dotenv
import argparse
import json
from datetime import datetime
import networkx as nx
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import config
from src import (
    KnowledgeGraphBuilder,
    GraphQuery,
    CustomRateLimiter
)
from src.agent import DocumentAnalysisAgent
from src.graph_builder_agent import GraphBuilderAgent

# Configure logging
logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class KnowledgeGraphPipeline:
    """Orchestrates the knowledge graph building and querying pipeline"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name=config.LLM.model_name,
            temperature=config.LLM.temperature
        )
        self.builder = KnowledgeGraphBuilder(api_key=api_key)
        self.rate_limiter = CustomRateLimiter(
            requests_per_minute=config.RATE_LIMITER.requests_per_minute,
            max_retries=config.RATE_LIMITER.max_retries
        )
        self.graph = None
        self.graph_query = None

    def build_graph(self, input_text: str, save_visualization: bool = True) -> None:
        """Build knowledge graph from input text"""
        logger.info("Starting knowledge graph construction")
        
        try:
            # Use GraphBuilderAgent for quality-focused construction
            builder_agent = GraphBuilderAgent(self.api_key)
            self.graph = builder_agent.build_quality_graph(input_text)
            
            # Save and visualize
            self._save_graph()
            if save_visualization:
                self.builder.visualize(save_path=config.VISUALIZATION_PATH)
                
            # Initialize graph query with LLM
            self.graph_query = GraphQuery(self.graph, self.llm)
            
            logger.info("Knowledge graph construction completed successfully")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise
            
    def _save_graph(self) -> None:
        """Save the graph to file"""
        if self.graph:
            # Save as GML
            nx.write_gml(self.graph, config.GRAPH_SAVE_PATH)
            
            # Save graph statistics
            stats = {
                'timestamp': datetime.now().isoformat(),
                'num_nodes': len(self.graph.nodes()),
                'num_edges': len(self.graph.edges()),
                'node_types': list(set(data.get('type', 'unknown') 
                                     for _, data in self.graph.nodes(data=True))),
                'edge_types': list(set(data.get('type', 'unknown') 
                                     for _, _, data in self.graph.edges(data=True)))
            }
            
            stats_file = Path('graph_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
    def query_graph(self, query: str) -> List[Document]:
        """Query the knowledge graph using the graph query"""
        if not self.graph_query:
            if os.path.exists(config.GRAPH_SAVE_PATH):
                self.graph = nx.read_gml(config.GRAPH_SAVE_PATH)
                self.graph_query = GraphQuery(self.graph, self.llm)
            else:
                raise ValueError("Graph must be built before querying")
            
        try:
            with self.rate_limiter:
                documents = self.graph_query.get_relevant_documents(query)
            return documents
        except Exception as e:
            logger.error(f"Error querying graph: {str(e)}")
            raise

    def run_interactive(self) -> None:
        """Run interactive document analysis session"""
        if not self.graph_query:
            if os.path.exists(config.GRAPH_SAVE_PATH):
                self.graph = nx.read_gml(config.GRAPH_SAVE_PATH)
                self.graph_query = GraphQuery(self.graph, self.llm)
            else:
                raise ValueError("Graph must be built before starting interactive session")
            
        try:
            agent = DocumentAnalysisAgent(self.graph_query)
            agent.run_interactive()
        except Exception as e:
            logger.error(f"Error in interactive session: {str(e)}")
            raise

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Knowledge Graph Builder and Query Tool')
    parser.add_argument('--input', '-i', type=str, help='Input text file path')
    parser.add_argument('--query', '-q', type=str, help='Query to run against the graph')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Update config if debug mode is enabled
    if args.debug:
        config.DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        pipeline = KnowledgeGraphPipeline(api_key)
        
        if args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                input_text = f.read()
            pipeline.build_graph(input_text, save_visualization=not args.no_viz)
            logger.info(f"Graph built and saved to {config.GRAPH_SAVE_PATH}")
            
            # Start interactive session after building graph
            pipeline.run_interactive()
            
        elif args.query:
            if not pipeline.graph:
                if os.path.exists(config.GRAPH_SAVE_PATH):
                    pipeline.graph = nx.read_gml(config.GRAPH_SAVE_PATH)
                    pipeline.graph_query = GraphQuery(pipeline.graph, pipeline.llm)
                else:
                    raise ValueError("No graph available for querying")
                    
            documents = pipeline.query_graph(args.query)
            print("\nQuery Results:")
            print("=" * 50)
            for i, doc in enumerate(documents, 1):
                print(f"\nDocument {i}:")
                print("-" * 30)
                print(doc)
                
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 