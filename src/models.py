from pydantic import BaseModel
from typing import List
from src.knowledge_graph import Entity, Relationship  # Import from knowledge_graph.py

class KnowledgeGraphData(BaseModel):
    """Complete Knowledge Graph data"""
    entities: List[Entity]
    relationships: List[Relationship] 