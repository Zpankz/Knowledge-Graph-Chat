from typing import List, Dict
import networkx as nx
from langchain_core.documents import Document
import numpy as np
import community.community_louvain as community_detection
import logging
from config import config
from src.llm import embeddings
from src.graph_cot import GraphCoT
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

class GraphQuery:
    """Enhanced graph querying with narrative-aware community detection"""
    
    def __init__(self, graph: nx.MultiDiGraph, llm: ChatGroq):
        self.graph = graph
        self.llm = llm
        self.communities = self._detect_narrative_communities()
        logger.info(f"Detected {len(set(self.communities.values()))} narrative communities")
        
    def _detect_narrative_communities(self) -> Dict[str, int]:
        """Detect communities with narrative coherence"""
        # First pass: structural communities
        base_communities = community_detection.best_partition(self.graph.to_undirected())
        
        # Get embeddings for semantic similarity
        node_embeddings = {}
        for node, data in self.graph.nodes(data=True):
            if 'embedding' in data:
                node_embeddings[node] = data['embedding']
        
        # Merge communities based on narrative similarity
        merged_communities = {}
        community_embeddings = {}
        
        # Calculate community centroids
        for node, comm_id in base_communities.items():
            if node in node_embeddings:
                if comm_id not in community_embeddings:
                    community_embeddings[comm_id] = []
                community_embeddings[comm_id].append(node_embeddings[node])
        
        # Calculate centroids
        community_centroids = {
            comm_id: np.mean(embeddings, axis=0)
            for comm_id, embeddings in community_embeddings.items()
            if embeddings
        }
        
        # Merge similar communities
        merged_id = 0
        processed = set()
        for comm1 in community_centroids:
            if comm1 in processed:
                continue
                
            similar_comms = [comm1]
            for comm2 in community_centroids:
                if comm2 != comm1 and comm2 not in processed:
                    similarity = np.dot(
                        community_centroids[comm1],
                        community_centroids[comm2]
                    ) / (
                        np.linalg.norm(community_centroids[comm1]) *
                        np.linalg.norm(community_centroids[comm2])
                    )
                    if similarity > config.RAG.similarity_threshold:
                        similar_comms.append(comm2)
            
            # Merge communities
            for node, comm_id in base_communities.items():
                if comm_id in similar_comms:
                    merged_communities[node] = merged_id
            
            processed.update(similar_comms)
            merged_id += 1
        
        logger.info(f"Merged {len(base_communities)} base communities into "
                   f"{len(set(merged_communities.values()))} narrative communities")
        
        return merged_communities
        
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
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using Chain of Thought analysis"""
        try:
            # Use GraphCoT for analysis
            cot = GraphCoT(self.graph, self.llm)
            relevant_info = cot.analyze_for_query(query)
            
            # Convert findings to documents
            documents = []
            for info in relevant_info:
                walk = info["walk"]
                analysis = info["analysis"]
                
                # Create document from walk insights
                steps_text = []
                if isinstance(walk, dict) and "steps" in walk:
                    # Handle dictionary case
                    for step in walk["steps"]:
                        if isinstance(step, dict):
                            step_text = f"- {step.get('current_node', 'unknown')}: {step.get('observation', '')}"
                            steps_text.append(step_text)
                elif hasattr(walk, 'steps'):
                    # Handle RandomWalk object case
                    for step in walk.steps:
                        step_text = f"- {step.current_node}: {step.observation}"
                        steps_text.append(step_text)
                
                doc = Document(
                    page_content=f"Walk Analysis:\n{analysis}\n\nPath Taken:\n" + 
                                "\n".join(steps_text),
                    metadata={
                        "start_node": walk.get("starting_node", "") if isinstance(walk, dict) else walk.starting_node,
                        "path_length": len(steps_text),
                        "conclusion": walk.get("conclusion", "") if isinstance(walk, dict) else walk.conclusion
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {str(e)}")
            # Return empty list on error
            return []
        
    def get_relevant_subgraphs(self, query_embedding: List[float],
                              node_embeddings: Dict[str, List[float]], 
                              top_k: int = 2) -> List[nx.MultiDiGraph]:
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
        """Generate a narrative summary of the subgraph"""
        # Get all text content
        chunk_texts = []
        for node, data in subgraph.nodes(data=True):
            if data.get('type') == 'Chunk':
                text = data.get('text', '')
                if text:
                    chunk_texts.append(text)
        
        # Get all entities and their relationships
        entities = {}
        relationships = []
        for node, data in subgraph.nodes(data=True):
            if data.get('type') == 'Entity':
                entities[node] = {
                    'label': data.get('label', ''),
                    'type': data.get('type', ''),
                    'properties': data.get('properties', {})
                }
        
        for u, v, data in subgraph.edges(data=True):
            if u in entities and v in entities:
                relationships.append({
                    'source': entities[u]['label'],
                    'target': entities[v]['label'],
                    'type': data.get('type', '')
                })
        
        # Combine into narrative summary
        summary = []
        
        # Add text content
        if chunk_texts:
            summary.append("Text Content:")
            summary.extend(chunk_texts)
        
        # Add entity information
        if entities:
            summary.append("\nKey Entities:")
            for entity_id, entity_data in entities.items():
                summary.append(f"- {entity_data['label']} ({entity_data['type']})")
                if entity_data['properties']:
                    for key, value in entity_data['properties'].items():
                        summary.append(f"  {key}: {value}")
        
        # Add relationship information
        if relationships:
            summary.append("\nRelationships:")
            for rel in relationships:
                summary.append(f"- {rel['source']} {rel['type']} {rel['target']}")
        
        return "\n".join(summary)