import os
import logging
from typing import Dict
import networkx as nx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.knowledge_graph import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

class FaustKGGenerator:
    def __init__(self):
        self.llm = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize necessary components"""
        try:
            load_dotenv()
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
                
            self.llm = ChatGroq(
                api_key=api_key,
                model_name="mixtral-8x7b-32768",
                temperature=0.3
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
            
    async def process_text(self, text: str) -> Dict:
        """Process text and generate knowledge graph asynchronously"""
        try:
            # Create knowledge graph
            builder = KnowledgeGraphBuilder(api_key=os.getenv('GROQ_API_KEY'))
            graph = builder.build(text)
            
            # Get statistics
            stats = {
                "nodes": len(graph.nodes()),
                "edges": len(graph.edges()),
                "density": nx.density(graph)
            }
            
            # Create visualization
            builder.visualize(save_path="temp_graph.png")
            
            return {
                "graph": graph,
                "statistics": stats,
                "visualization": "temp_graph.png"
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise 