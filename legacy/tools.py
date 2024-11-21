from typing import Dict, List, Any
from pydantic import BaseModel, Field
from outlines import models, prompts
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from io import StringIO, BytesIO
import networkx as nx
import json
import logging

class TextProcessingTool(BaseModel):
    """Tool for processing text using docling"""
    name: str = Field("text_processor", description="Tool for text processing")
    description: str = Field("Process and structure text for better LLM understanding")
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text using docling document converter"""
        # Create document converter
        converter = DocumentConverter()
        
        # Convert text to bytes and create BytesIO stream
        text_bytes = text.encode('utf-8')
        bytes_stream = BytesIO(text_bytes)
        
        # Create document stream with required fields
        doc_stream = DocumentStream(
            name="input.txt",  # Changed from filename to name
            stream=bytes_stream,  # Using BytesIO instead of StringIO
            mime_type="text/plain"  # Added required mime_type
        )
        
        # Process the document
        try:
            result = converter.convert(doc_stream)
            
            # Extract structured information
            processed_data = {
                "text": result.document.text,
                "metadata": result.document.metadata,
                "sections": [
                    {
                        "text": section.text,
                        "type": section.type,
                        "metadata": section.metadata
                    }
                    for section in result.document.sections
                ],
                "paragraphs": [
                    {
                        "text": para.text,
                        "metadata": para.metadata
                    }
                    for para in result.document.paragraphs
                ]
            }
            
            return processed_data
            
        except Exception as e:
            raise ValueError(f"Error processing text with Docling: {str(e)}")

class KnowledgeGraphTool(BaseModel):
    """Tool for knowledge graph extraction using outlines"""
    name: str = Field("kg_extractor", description="Tool for knowledge graph extraction")
    description: str = Field("Extract knowledge graph from text using outlines")
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract knowledge graph using outlines"""
        try:
            # Define the schema
            schema = {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "name": {"type": "string"},
                                "properties": {"type": "object"}
                            },
                            "required": ["id", "type", "name"]
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "type": {"type": "string"},
                                "evidence": {"type": "string"}
                            },
                            "required": ["source", "target", "type"]
                        }
                    }
                },
                "required": ["entities", "relationships"]
            }

            # Create the prompt
            prompt = f"""
            Extract a knowledge graph from this text. Identify entities and their relationships.
            Format the output as JSON with the following structure:
            {{
                "entities": [
                    {{"id": "unique_id", "type": "entity_type", "name": "entity_name", "properties": {{}}}}
                ],
                "relationships": [
                    {{"source": "entity_id", "target": "entity_id", "type": "relationship_type", "evidence": "text evidence"}}
                ]
            }}

            Text to analyze: {text}
            """

            # Generate the structured output
            result = self.llm.invoke(prompt)
            
            # Parse the JSON response
            try:
                extracted_data = json.loads(result)
                return extracted_data
            except json.JSONDecodeError:
                logging.error("Failed to parse LLM response as JSON")
                return {"entities": [], "relationships": []}
            
        except Exception as e:
            logging.warning(f"Error in knowledge graph extraction: {str(e)}")
            return {
                "entities": [],
                "relationships": []
            }

class GraphAnalysisTool(BaseModel):
    """Tool for analyzing and validating knowledge graphs"""
    name: str = Field("graph_analyzer", description="Tool for graph analysis")
    description: str = Field("Analyze and validate graph structure")
    
    def analyze(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze graph structure and properties with empty graph handling"""
        if len(graph.nodes()) == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
                "is_connected": False,
                "centrality": {},
                "status": "empty_graph",
                "recommendations": [
                    "Add more specific entity extraction",
                    "Check text segmentation",
                    "Verify relationship extraction"
                ]
            }
            
        try:
            analysis = {
                "node_count": len(graph.nodes()),
                "edge_count": len(graph.edges()),
                "density": nx.density(graph),
                "is_connected": nx.is_weakly_connected(graph) if len(graph.nodes()) > 1 else False,
                "centrality": {
                    node: score 
                    for node, score in nx.degree_centrality(graph).items()
                },
                "status": "analyzed",
                "metrics": {
                    "avg_degree": sum(dict(graph.degree()).values()) / len(graph.nodes()),
                    "clustering": nx.average_clustering(graph.to_undirected()) if len(graph.nodes()) > 2 else 0.0
                }
            }
            return analysis
            
        except Exception as e:
            logging.error(f"Error in graph analysis: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
                "node_count": len(graph.nodes()),
                "edge_count": len(graph.edges())
            }