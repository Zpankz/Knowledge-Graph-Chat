from typing import Dict, List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
import networkx as nx
import json
import logging
from llm import embeddings  # Import the local embeddings model
import numpy as np

logger = logging.getLogger(__name__)

class GraphRetriever(BaseRetriever):
    """Graph-based retriever for knowledge graph querying"""
    
    class Config:
        arbitrary_types_allowed = True
    
    # Define fields with proper types and defaults
    graph: nx.MultiDiGraph = Field(description="The knowledge graph")
    llm: Any = Field(description="The language model")
    node_types: List[str] = Field(default_factory=list)
    edge_types: List[str] = Field(default_factory=list)
    property_keys: List[str] = Field(default_factory=list)
    chunks: Dict[str, str] = Field(default_factory=dict)
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    last_relevant_nodes: List[str] = Field(default_factory=list)

    def __init__(self, graph: nx.MultiDiGraph, llm: Any):
        """Initialize the graph retriever with a graph and LLM"""
        # Initialize with required fields
        super().__init__(graph=graph, llm=llm)
        
        # Extract schema information from graph
        self.node_types = self._extract_node_types()
        self.edge_types = self._extract_edge_types()
        self.property_keys = self._extract_property_keys()
        
        # Initialize chunks and embeddings
        self._initialize_chunks()
        
        logger.info(f"Initialized GraphRetriever with {len(self.node_types)} node types, "
                   f"{len(self.edge_types)} edge types, and {len(self.chunks)} text chunks")

    def _initialize_chunks(self):
        """Initialize text chunks and their embeddings"""
        # Extract text from all node types
        for node, data in self.graph.nodes(data=True):
            # Collect all text content associated with the node
            text_parts = []
            
            # Add node ID and type
            text_parts.append(f"{node} ({data.get('type', 'Unknown')})")
            
            # Add properties
            properties = data.get('properties', {})
            if properties:
                for key, value in properties.items():
                    text_parts.append(f"{key}: {value}")
            
            # Add dialogue content if available
            if data.get('type') == 'Dialogue':
                text_parts.append(f"Speaker: {properties.get('speaker', '')}")
                text_parts.append(f"Text: {properties.get('text', '')}")
            
            # Add scene content if available
            if data.get('type') == 'Scene':
                text_parts.append(f"Scene Description: {properties.get('description', '')}")
                text_parts.append(f"Location: {properties.get('location', '')}")
            
            # Combine all text
            text = "\n".join(text_parts)
            
            if text:
                logger.info(f"Adding text chunk for node {node}: {text[:100]}...")
                self.chunks[node] = text
                try:
                    self.chunk_embeddings[node] = embeddings.embed_documents([text])[0]
                except Exception as e:
                    logger.error(f"Error embedding text for node {node}: {str(e)}")

    def _extract_node_types(self) -> List[str]:
        """Extract all unique node types from the graph"""
        return list(set(
            data.get('type', 'Unknown') 
            for _, data in self.graph.nodes(data=True)
        ))

    def _extract_edge_types(self) -> List[str]:
        """Extract all unique edge types from the graph"""
        return list(set(
            data.get('type', 'RELATED_TO') 
            for _, _, data in self.graph.edges(data=True)
        ))

    def _extract_property_keys(self) -> List[str]:
        """Extract all unique property keys from nodes and edges"""
        property_keys = set()
        
        # Node properties
        for _, data in self.graph.nodes(data=True):
            properties = data.get('properties', {})
            property_keys.update(properties.keys())
            
        # Edge properties
        for _, _, data in self.graph.edges(data=True):
            properties = data.get('properties', {})
            property_keys.update(properties.keys())
            
        return list(property_keys)

    def _get_relevant_nodes(self, query: str, top_k: int = 3) -> List[str]:
        """Get relevant nodes based on query embedding similarity"""
        try:
            # Get query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Calculate similarities with all chunks
            similarities = {}
            for node_id, node_embedding in self.chunk_embeddings.items():
                similarity = self._cosine_similarity(query_embedding, node_embedding)
                similarities[node_id] = similarity
            
            # Sort by similarity and get top k
            sorted_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            relevant_nodes = [node for node, _ in sorted_nodes[:top_k]]
            
            # Store for visualization
            self.last_relevant_nodes = relevant_nodes
            
            # If no relevant nodes found, return most central nodes
            if not relevant_nodes:
                centrality = nx.degree_centrality(self.graph)
                relevant_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
                relevant_nodes = [node for node, _ in relevant_nodes]
            
            return relevant_nodes
            
        except Exception as e:
            logger.error(f"Error getting relevant nodes: {str(e)}")
            return []

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on the query"""
        try:
            relevant_nodes = self._get_relevant_nodes(query)
            documents = []
            
            for node in relevant_nodes:
                # Create subgraph centered on this node
                subgraph = nx.ego_graph(self.graph, node, radius=2)
                
                # Convert node and its context to text
                text = self._node_to_text(node, subgraph)
                
                # Add source information
                metadata = {
                    "node_id": node,
                    "node_type": self.graph.nodes[node].get('type', 'Unknown'),
                    "properties": self.graph.nodes[node].get('properties', {})
                }
                
                documents.append(Document(page_content=text, metadata=metadata))
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {str(e)}")
            return []

    def _node_to_text(self, node: str, subgraph: nx.MultiDiGraph) -> str:
        """Convert a node and its context to text representation"""
        try:
            node_data = self.graph.nodes[node]
            text_parts = []
            
            # Add node information with properties
            text_parts.append(f"Entity: {node_data.get('label', node)}")
            text_parts.append(f"Type: {node_data.get('type', 'Unknown')}")
            
            # Add properties with better formatting
            properties = node_data.get('properties', {})
            if properties:
                text_parts.append("Properties:")
                if properties.get('name'):
                    text_parts.append(f"  Name: {properties['name']}")
                if properties.get('attributes'):
                    text_parts.append(f"  Attributes: {', '.join(properties['attributes'])}")
                if properties.get('context'):
                    text_parts.append(f"  Context: {properties['context']}")
            
            # Add relationships with context
            relationships = []
            for source, target, data in subgraph.edges(data=True):
                rel_props = data.get('properties', {})
                if source == node:
                    target_data = subgraph.nodes[target]
                    rel_text = (
                        f"{node_data.get('label', source)} {data.get('type', 'relates to')} "
                        f"{target_data.get('label', target)}"
                    )
                    if rel_props.get('description'):
                        rel_text += f"\n    Description: {rel_props['description']}"
                    if rel_props.get('context'):
                        rel_text += f"\n    Context: {rel_props['context']}"
                    relationships.append(rel_text)
                elif target == node:
                    source_data = subgraph.nodes[source]
                    rel_text = (
                        f"{source_data.get('label', source)} {data.get('type', 'relates to')} "
                        f"{node_data.get('label', target)}"
                    )
                    if rel_props.get('description'):
                        rel_text += f"\n    Description: {rel_props['description']}"
                    if rel_props.get('context'):
                        rel_text += f"\n    Context: {rel_props['context']}"
                    relationships.append(rel_text)
            
            if relationships:
                text_parts.append("\nRelationships:")
                text_parts.extend(relationships)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error converting node to text: {str(e)}")
            return str(node)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        return self.get_relevant_documents(query)