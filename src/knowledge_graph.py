from typing import Dict, List, Any, Tuple, Set, Optional
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
import logging
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from io import BytesIO
import json
import community.community_louvain as community_detection
from src.llm import embeddings
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Entity(BaseModel):
    """Entity in the knowledge graph"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class Relationship(BaseModel):
    """Relationship between entities"""
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    """Knowledge graph structure"""
    entities: List[Entity]
    relationships: List[Relationship]

class KnowledgeGraphBuilder:
    def __init__(self, api_key: str):
        """Initialize the builder with API key"""
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=config.LLM.model_name,
            temperature=config.LLM.temperature
        )
        self.graph = nx.MultiDiGraph()
        logger.info("KnowledgeGraphBuilder initialized")

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into meaningful chunks"""
        try:
            # First try sentence-based chunking
            sentences = text.split('.')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip() + '.'
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                # If adding this sentence would exceed chunk size, save current chunk
                if current_length + len(sentence) > config.RAG.chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += len(sentence)
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # If chunking produced no results, fallback to simpler method
            if not chunks:
                chunks = [text[i:i + config.RAG.chunk_size] 
                         for i in range(0, len(text), config.RAG.chunk_size - config.RAG.chunk_overlap)]
            
            logger.info(f"Text preprocessed into {len(chunks)} chunks")
            
            # Ensure each chunk has meaningful content
            filtered_chunks = []
            for chunk in chunks:
                # Clean the chunk
                clean_chunk = chunk.strip()
                # Only keep chunks with sufficient content
                if len(clean_chunk.split()) >= 5:  # At least 5 words
                    filtered_chunks.append(clean_chunk)
            
            logger.info(f"After filtering: {len(filtered_chunks)} meaningful chunks")
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            # Fallback to paragraph splitting
            paragraphs = text.split('\n\n')
            meaningful_chunks = [p.strip() for p in paragraphs if len(p.strip().split()) >= 5]
            logger.info(f"Fallback: Text split into {len(meaningful_chunks)} chunks")
            return meaningful_chunks

    def extract_knowledge(self, chunk: str) -> KnowledgeGraph:
        """Extract basic knowledge graph from text chunk"""
        try:
            # Simple entity extraction
            words = chunk.split()
            entities = []
            entity_id = 1
            
            # Extract entities based on capitalization and length
            for word in words:
                word = word.strip('.,!?()[]{}":;')
                if (word and word[0].isupper() and len(word) > 2  # Avoid single letters
                    and not word.isupper()):  # Avoid ALL CAPS words
                    entity = Entity(
                        id=f"e{entity_id}",
                        label=word,
                        type=self._infer_entity_type(word, chunk),
                        properties={"context": chunk[:200]}  # Some context
                    )
                    entities.append(entity)
                    entity_id += 1
            
            # Create basic relationships between nearby entities
            relationships = []
            if len(entities) > 1:
                for i in range(len(entities)-1):
                    relationship = Relationship(
                        source=entities[i].id,
                        target=entities[i+1].id,
                        type="RELATES_TO",
                        properties={
                            "context": "sequential occurrence",
                            "distance": i+1
                        }
                    )
                    relationships.append(relationship)
            
            logger.info(f"Extracted {len(entities)} entities and "
                       f"{len(relationships)} relationships")
            
            return KnowledgeGraph(
                entities=entities,
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction: {str(e)}")
            return KnowledgeGraph(entities=[], relationships=[])

    def _infer_entity_type(self, word: str, context: str) -> str:
        """Simple entity type inference"""
        # Common name lists (could be expanded)
        person_indicators = ['Mr', 'Mrs', 'Ms', 'Dr', 'Professor', 'said', 'spoke', 'told']
        location_indicators = ['in', 'at', 'from', 'to', 'Street', 'Road', 'Avenue', 'City']
        event_indicators = ['meeting', 'conference', 'event', 'happened', 'occurred']
        
        # Check surrounding context (20 words before and after)
        words_before = ' '.join(context.split()[:20]).lower()
        words_after = ' '.join(context.split()[-20:]).lower()
        surrounding = words_before + ' ' + words_after
        
        # Simple type inference
        if any(indicator in surrounding for indicator in person_indicators):
            return "person"
        elif any(indicator in surrounding for indicator in location_indicators):
            return "location"
        elif any(indicator in surrounding for indicator in event_indicators):
            return "event"
        
        return "object"  # Default type

    def update_graph(self, knowledge: KnowledgeGraph):
        """Update the NetworkX graph with new knowledge"""
        try:
            # Add entities as nodes
            for entity in knowledge.entities:
                self.graph.add_node(
                    entity.id,
                    label=entity.label,
                    type=entity.type,
                    **entity.properties
                )

            # Add relationships as edges
            for rel in knowledge.relationships:
                if rel.source in self.graph and rel.target in self.graph:
                    self.graph.add_edge(
                        rel.source,
                        rel.target,
                        type=rel.type,
                        **rel.properties
                    )
            
            logger.info(f"Graph updated: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        except Exception as e:
            logger.error(f"Error updating graph: {str(e)}")
            raise

    def build(self, text: str) -> nx.MultiDiGraph:
        """Build knowledge graph with proper schema and relationships"""
        try:
            # 1. Create Document node
            doc_id = "doc_0"
            self.graph.add_node(doc_id, 
                              type="Document",
                              label="Story Document",
                              source="text_input",
                              content=text)  # Store full text

            # 2. Create and link Chunk nodes with embeddings
            chunks = self.preprocess_text(text)
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                # Create embedding for chunk
                chunk_embedding = embeddings.embed_query(chunk_text)
                self.graph.add_node(chunk_id,
                                  type="Chunk",
                                  label=f"Text Chunk {i}",
                                  text=chunk_text,
                                  embedding=chunk_embedding)
                # Link chunk to document
                self.graph.add_edge(chunk_id, doc_id, 
                                  type="PART_OF",
                                  relationship="chunk_of_document")

                # 3. Extract entities and relationships for each chunk
                knowledge = self.extract_knowledge(chunk_text)
                
                # Add entities with embeddings and chunk context
                for entity in knowledge.entities:
                    if entity.id not in self.graph:
                        # Create embedding for entity name and description
                        entity_text = f"{entity.label} {entity.properties.get('description', '')}"
                        entity_embedding = embeddings.embed_query(entity_text)
                        
                        self.graph.add_node(entity.id,
                                          type="Entity",
                                          label=entity.label,
                                          entity_type=entity.type,
                                          properties=entity.properties,
                                          embedding=entity_embedding)
                    # Link entity to chunk
                    self.graph.add_edge(chunk_id, entity.id, 
                                      type="HAS_ENTITY",
                                      relationship="mentioned_in_chunk")

                # Add relationships between entities
                for rel in knowledge.relationships:
                    if rel.source in self.graph and rel.target in self.graph:
                        self.graph.add_edge(rel.source, 
                                          rel.target,
                                          type="RELATES_TO",
                                          relationship=rel.type,
                                          properties=rel.properties)

            # 4. Create community structure with hierarchical relationships
            undirected_graph = self.graph.to_undirected()
            communities = community_detection.best_partition(undirected_graph)
            
            # Track parent communities for hierarchy
            parent_communities = {}
            
            # First level communities
            for community_id in set(communities.values()):
                community_members = [node for node, comm in communities.items() 
                                  if comm == community_id]
                
                # Get all text and entities in this community
                community_content = []
                community_entities = []
                for node in community_members:
                    node_data = self.graph.nodes[node]
                    if node_data.get('type') == 'Chunk':
                        community_content.append(node_data.get('text', ''))
                    elif node_data.get('type') == 'Entity':
                        community_entities.append(node_data)
                
                # Generate community summary
                full_content = "\n".join(community_content)
                summary_prompt = f"""Summarize this content and its key entities into a coherent narrative:
                
                Content: {full_content}
                
                Entities: {[e.get('label') for e in community_entities]}
                """
                summary = self.llm.invoke(summary_prompt).content
                
                # Create community node
                comm_node_id = f"community_{community_id}"
                self.graph.add_node(comm_node_id,
                                  type="Community",
                                  label=f"Story Community {community_id}",
                                  level=1,
                                  summary=summary,
                                  full_content=full_content,
                                  weight=len(community_members))
                
                # Link members to community
                for member in community_members:
                    self.graph.add_edge(member, comm_node_id, 
                                      type="IN_COMMUNITY",
                                      relationship="member_of_community")
                
                parent_communities[community_id] = comm_node_id
            
            # Create higher-level communities if needed
            if len(parent_communities) > 3:  # Create parent communities for better organization
                parent_comm_id = "parent_community_0"
                self.graph.add_node(parent_comm_id,
                                  type="Community",
                                  label="Main Story",
                                  level=2,
                                  summary="Overall story narrative",
                                  weight=len(parent_communities))
                
                # Link child communities to parent
                for child_id in parent_communities.values():
                    self.graph.add_edge(child_id, parent_comm_id,
                                      type="PARENT_COMMUNITY",
                                      relationship="part_of_larger_narrative")

            logger.info(f"Built enhanced graph with {len(self.graph.nodes())} nodes and "
                       f"{len(self.graph.edges())} edges")
            return self.graph

        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise

    def visualize(self, save_path: str) -> None:
        """Visualize the knowledge graph"""
        try:
            plt.figure(figsize=config.GRAPH.figure_size)
            pos = nx.spring_layout(self.graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=1500)
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
            
            # Add labels
            labels = {node: f"{data.get('label', node)}\n({data.get('type', 'unknown')})"
                     for node, data in self.graph.nodes(data=True)}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            # Add edge labels
            edge_labels = {(u, v): data.get('type', '')
                         for u, v, data in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
            
            plt.title("Knowledge Graph")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Graph visualization saved to {save_path}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            raise