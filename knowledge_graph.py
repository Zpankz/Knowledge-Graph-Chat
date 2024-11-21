from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
import logging
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from io import BytesIO, StringIO
import json

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
            model_name="mixtral-8x7b-32768",
            temperature=0.3
        )
        self.graph = nx.MultiDiGraph()
        logger.info("KnowledgeGraphBuilder initialized")

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into chunks using docling"""
        try:
            converter = DocumentConverter()
            byte_stream = BytesIO(text.encode('utf-8'))
            doc_stream = DocumentStream(
                name="input.txt",
                stream=byte_stream,
                mime_type="text/plain"
            )
            result = converter.convert(doc_stream)
            chunks = [para.text for para in result.document.paragraphs]
            logger.info(f"Text preprocessed into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            chunks = text.split('\n\n')
            logger.info(f"Fallback: Text split into {len(chunks)} chunks")
            return [chunk.strip() for chunk in chunks if chunk.strip()]

    def extract_knowledge(self, chunk: str) -> KnowledgeGraph:
        """Extract knowledge graph from text chunk"""
        try:
            prompt = """Extract a knowledge graph from this text. Create meaningful entities and relationships.
            Return a JSON object with this exact structure:
            {
                "entities": [
                    {
                        "id": "e1",
                        "label": "entity name",
                        "type": "entity type",
                        "properties": {
                            "name": "full name or description",
                            "attributes": ["relevant attributes"],
                            "context": "contextual information"
                        }
                    }
                ],
                "relationships": [
                    {
                        "source": "e1",
                        "target": "e2", 
                        "type": "relationship type",
                        "properties": {
                            "description": "detailed description of relationship",
                            "context": "when/where this relationship occurs"
                        }
                    }
                ]
            }
            
            Rules:
            1. Each entity must have a meaningful label and descriptive properties
            2. Relationships should capture the narrative flow with context
            3. Entity types: person, location, object, action, event, etc.
            4. Relationship types should be descriptive: owns, visits, gives_to, speaks_to, etc.
            5. Include relevant context and attributes in properties

            Text: """ + chunk

            response = self.llm.invoke(prompt)
            
            try:
                content = response.content
                if isinstance(content, str):
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        result = json.loads(json_str)
                        
                        if not isinstance(result, dict):
                            raise ValueError("Response is not a dictionary")
                        
                        entities = result.get("entities", [])
                        relationships = result.get("relationships", [])
                        
                        clean_entities = []
                        for e in entities:
                            if all(k in e for k in ["id", "label", "type"]):
                                # Include all properties for richer context
                                clean_entities.append(Entity(
                                    id=str(e["id"]),
                                    label=str(e["label"]),
                                    type=str(e["type"]),
                                    properties=e.get("properties", {})
                                ))
                        
                        clean_relationships = []
                        for r in relationships:
                            if all(k in r for k in ["source", "target", "type"]):
                                # Include relationship properties for context
                                clean_relationships.append(Relationship(
                                    source=str(r["source"]),
                                    target=str(r["target"]),
                                    type=str(r["type"]),
                                    properties=r.get("properties", {})
                                ))
                        
                        return KnowledgeGraph(
                            entities=clean_entities,
                            relationships=clean_relationships
                        )
                
                raise ValueError("Invalid response format")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return KnowledgeGraph(entities=[], relationships=[])
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {str(e)}")
            return KnowledgeGraph(entities=[], relationships=[])

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

    def build(self, text: str):
        """Build knowledge graph from text"""
        try:
            # Preprocess text into chunks
            chunks = self.preprocess_text(text)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                knowledge = self.extract_knowledge(chunk)
                self.update_graph(knowledge)
            
            return self.graph
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise

    def visualize(self, save_path: Optional[str] = None):
        """Visualize the knowledge graph"""
        try:
            plt.figure(figsize=(12, 8))
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