from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Entity(BaseModel):
    """Entity in the Knowledge Graph"""
    id: str = Field(..., description="Unique identifier of the entity")
    label: str = Field(..., description="Label/name of the entity")
    type: str = Field(..., description="Type of entity (Person, Location, etc.)")
    properties: Dict = Field(default_factory=dict, description="Additional properties")
    confidence: float = Field(default=1.0, description="Confidence score")

class Relationship(BaseModel):
    """Relationship in the Knowledge Graph"""
    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Type of relationship")
    properties: Dict = Field(default_factory=dict, description="Additional properties")
    confidence: float = Field(default=1.0, description="Confidence score")
    evidence: str = Field(..., description="Text evidence supporting this relationship")

class KnowledgeGraphData(BaseModel):
    """Complete Knowledge Graph data"""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list) 