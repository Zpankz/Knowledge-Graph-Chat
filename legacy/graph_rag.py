from typing import Dict, List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
import networkx as nx
import json
import logging
from src.llm import embeddings  # Import the local embeddings model
import numpy as np
from community import community_louvain
from config import config

logger = logging.getLogger(__name__)

class GraphQuery:
    """Enhanced graph querying with community detection and dynamic subgraph selection"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        # Detect communities on initialization
        self.communities = self._detect_communities()
        logger.info(f"Detected {len(set(self.communities.values()))} communities in graph")
        
    def _detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain method"""
        return community_louvain.best_partition(self.graph.to_undirected())
        
    def _get_community_relevance(self, query_embedding: List[float], 
                               node_embeddings: Dict[str, List[float]]) -> Dict[int, float]:
        """Calculate relevance scores for each community"""
        community_scores = {}
        
        for community_id in set(self.communities.values()):
            # Get nodes in this community
            community_nodes = [node for node, comm in self.communities.items() 
                             if comm == community_id]
            
            # Calculate average embedding for community
            community_embeddings = [node_embeddings[node] for node in community_nodes 
                                  if node in node_embeddings]
            if not community_embeddings:
                continue
                
            community_centroid = np.mean(community_embeddings, axis=0)
            
            # Calculate similarity with query
            similarity = np.dot(query_embedding, community_centroid) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(community_centroid)
            )
            community_scores[community_id] = similarity
            
        return community_scores
        
    def get_relevant_subgraphs(self, query_embedding: List[float],
                              node_embeddings: Dict[str, List[float]], 
                              top_k: int = config.RAG.top_k_communities) -> List[nx.MultiDiGraph]:
        """Get most relevant subgraphs based on community detection"""
        # Get community relevance scores
        community_scores = self._get_community_relevance(query_embedding, node_embeddings)
        
        # Get top-k communities
        top_communities = sorted(community_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:top_k]
        
        subgraphs = []
        for community_id, score in top_communities:
            # Get nodes in this community
            community_nodes = [node for node, comm in self.communities.items() 
                             if comm == community_id]
            
            # Extract subgraph
            subgraph = self.graph.subgraph(community_nodes)
            subgraphs.append(subgraph)
            
            logger.info(f"Selected community {community_id} with {len(community_nodes)} nodes "
                       f"and relevance score {score:.3f}")
            
        return subgraphs
        
    def summarize_subgraph(self, subgraph: nx.MultiDiGraph) -> str:
        """Generate a summary of the subgraph"""
        summary_parts = []
        
        # Get central nodes
        centrality = nx.degree_centrality(subgraph)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for node, centrality_score in top_nodes:
            node_data = subgraph.nodes[node]
            
            # Node information
            summary_parts.append(f"Entity: {node_data.get('label', node)}")
            summary_parts.append(f"Type: {node_data.get('type', 'Unknown')}")
            
            # Key relationships
            relationships = []
            for _, target, data in subgraph.out_edges(node, data=True):
                target_data = subgraph.nodes[target]
                rel = (f"{node_data.get('label', node)} {data.get('type', 'relates to')} "
                      f"{target_data.get('label', target)}")
                relationships.append(rel)
            
            if relationships:
                summary_parts.append("Key relationships:")
                summary_parts.extend(f"- {rel}" for rel in relationships[:3])
            
            summary_parts.append("")
            
        return "\n".join(summary_parts)

class GraphRetriever(BaseRetriever):
    """Graph-based retriever for knowledge graph querying"""
    
    class Config:
        arbitrary_types_allowed = True
    
    # Define fields with proper types and defaults
    graph: nx.MultiDiGraph = Field(description="The knowledge graph")
    llm: Any = Field(description="The language model")
    graph_query: GraphQuery = Field(description="Graph query processor")
    node_types: List[str] = Field(default_factory=list)
    edge_types: List[str] = Field(default_factory=list)
    property_keys: List[str] = Field(default_factory=list)
    chunks: Dict[str, str] = Field(default_factory=dict)
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    last_relevant_nodes: List[str] = Field(default_factory=list)
    node_embeddings: Dict[str, List[float]] = Field(default_factory=dict)

    def __init__(self, graph: nx.MultiDiGraph, llm: Any):
        """Initialize the graph retriever with a graph and LLM"""
        # Initialize base fields first
        super().__init__(
            graph=graph, 
            llm=llm,
            graph_query=GraphQuery(graph),
            node_embeddings={}
        )
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize node embeddings"""
        for node, data in self.graph.nodes(data=True):
            text = self._node_to_text(node, self.graph)
            if text:
                self.node_embeddings[node] = embeddings.embed_query(text)
                
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using community-aware retrieval"""
        try:
            # Get query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Get relevant subgraphs
            relevant_subgraphs = self.graph_query.get_relevant_subgraphs(
                query_embedding,
                self.node_embeddings
            )
            
            documents = []
            for subgraph in relevant_subgraphs:
                # Get subgraph summary
                summary = self.graph_query.summarize_subgraph(subgraph)
                
                # Get central nodes for metadata
                centrality = nx.degree_centrality(subgraph)
                central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                central_node_info = []
                
                for node, _ in central_nodes:
                    node_data = subgraph.nodes[node]
                    central_node_info.append({
                        'id': node,
                        'label': node_data.get('label', node),
                        'type': node_data.get('type', 'Unknown')
                    })
                
                # Add as document with enhanced metadata
                metadata = {
                    "node_count": len(subgraph.nodes()),
                    "edge_count": len(subgraph.edges()),
                    "community_id": next(iter(
                        self.graph_query.communities[node] 
                        for node in subgraph.nodes()
                    )),
                    "central_nodes": central_node_info  # Add central nodes info
                }
                
                documents.append(Document(
                    page_content=summary,
                    metadata=metadata
                ))
            
            logger.info(f"Retrieved {len(documents)} community-based documents")
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