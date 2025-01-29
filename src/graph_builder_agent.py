"""Agent for building and improving knowledge graph schema and ontology"""
from typing import Annotated, TypedDict, List, Dict, Any
from dataclasses import dataclass
import logging
import os
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, BaseMessage
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
import networkx as nx
import numpy as np
import json

from src.knowledge_graph import KnowledgeGraphBuilder
from config import config
from src.graph_cot import GraphCoT

logger = logging.getLogger(__name__)

@dataclass
class OntologyMetrics:
    """Metrics for evaluating knowledge graph ontology quality"""
    schema_coverage: float = 0.0    # How well entities match defined types
    relation_precision: float = 0.0  # How specific and accurate relationships are
    semantic_consistency: float = 0.0 # How well relations follow domain rules
    context_completeness: float = 0.0 # How well context is captured
    
    def average_score(self) -> float:
        metrics = [self.schema_coverage, self.relation_precision,
                  self.semantic_consistency, self.context_completeness]
        return sum(metrics) / len(metrics)

class GraphBuilderState(TypedDict):
    """State for the graph builder agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    text: str
    graph: nx.MultiDiGraph
    metrics: OntologyMetrics
    schema: Dict[str, Any]  # Current schema definition
    iteration: int

class GraphBuilderAgent:
    """Agent for iteratively improving knowledge graph schema and ontology"""
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.3
        )
        
        # Initialize with base schema
        self.base_schema = {
            "entity_types": {
                "Document": {
                    "required_attributes": ["content", "title"],
                    "optional_attributes": ["metadata"]
                },
                "Chunk": {
                    "required_attributes": ["text", "embedding"],
                    "optional_attributes": ["position"]
                },
                "Entity": {
                    "required_attributes": ["label", "type"],
                    "optional_attributes": ["properties"]
                },
                "Community": {
                    "required_attributes": ["label"],
                    "optional_attributes": ["summary", "members"]
                }
            },
            "relation_types": {
                "CONTAINS": {
                    "valid_sources": ["Document", "Chunk"],
                    "valid_targets": ["Chunk", "Entity"],
                    "properties": ["position"]
                },
                "PART_OF": {
                    "valid_sources": ["Chunk", "Entity"],
                    "valid_targets": ["Document", "Community"],
                    "properties": ["confidence"]
                },
                "RELATES_TO": {
                    "valid_sources": ["Entity"],
                    "valid_targets": ["Entity"],
                    "properties": ["relationship_type", "strength"]
                },
                "IN_COMMUNITY": {
                    "valid_sources": ["Entity", "Chunk"],
                    "valid_targets": ["Community"],
                    "properties": ["membership_strength"]
                }
            },
            "constraints": [
                "Each Chunk must belong to exactly one Document",
                "Entities must have unique identifiers",
                "Communities must have at least one member"
            ]
        }
        
        # Tools for schema and ontology improvement
        self.tools = [
            Tool(
                name="analyze_schema",
                description="Analyze current schema coverage and consistency",
                func=self._analyze_schema,
            ),
            Tool(
                name="extract_ontology",
                description="Extract domain ontology from text",
                func=self._extract_ontology,
            ),
            Tool(
                name="improve_relations",
                description="Improve relationship types and semantics",
                func=self._improve_relations,
            ),
            Tool(
                name="enhance_context",
                description="Enhance contextual information",
                func=self._enhance_context,
            )
        ]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _analyze_schema(self, state: GraphBuilderState) -> Dict[str, Any]:
        """Analyze schema coverage and consistency"""
        try:
            schema = state["schema"]
            graph = state["graph"]
            
            # Calculate metrics
            metrics = {
                "schema_coverage": 0.0,
                "relation_precision": 0.0,
                "semantic_consistency": 0.0,
                "context_completeness": 0.0
            }
            
            # 1. Schema Coverage
            entity_types = set(data.get('type', '') for _, data in graph.nodes(data=True))
            valid_types = set(schema.get('entity_types', {}).keys())
            if valid_types:
                metrics["schema_coverage"] = len(entity_types.intersection(valid_types)) / len(valid_types)
            
            # 2. Relation Precision
            relation_types = set(data.get('type', '') for _, _, data in graph.edges(data=True))
            valid_relations = set(schema.get('relation_types', {}).keys())
            if valid_relations:
                metrics["relation_precision"] = len(relation_types.intersection(valid_relations)) / len(valid_relations)
            
            # 3. Semantic Consistency
            valid_edges = 0
            total_edges = 0
            for u, v, data in graph.edges(data=True):
                total_edges += 1
                rel_type = data.get('type', '')
                if rel_type in schema.get('relation_types', {}):
                    rel_schema = schema['relation_types'][rel_type]
                    u_type = graph.nodes[u].get('type', '')
                    v_type = graph.nodes[v].get('type', '')
                    if (u_type in rel_schema.get('valid_sources', []) and 
                        v_type in rel_schema.get('valid_targets', [])):
                        valid_edges += 1
            
            metrics["semantic_consistency"] = valid_edges / total_edges if total_edges > 0 else 0.0
            
            # 4. Context Completeness
            complete_nodes = 0
            total_nodes = 0
            for _, data in graph.nodes(data=True):
                node_type = data.get('type', '')
                if node_type in schema.get('entity_types', {}):
                    total_nodes += 1
                    required_attrs = schema['entity_types'][node_type].get('required_attributes', [])
                    if all(attr in data for attr in required_attrs):
                        complete_nodes += 1
            
            metrics["context_completeness"] = complete_nodes / total_nodes if total_nodes > 0 else 0.0
            
            logger.info(f"Schema Analysis Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing schema: {str(e)}")
            return {
                "schema_coverage": 0.0,
                "relation_precision": 0.0,
                "semantic_consistency": 0.0,
                "context_completeness": 0.0
            }

    def _extract_ontology(self, text: str) -> Dict[str, Any]:
        """Extract domain ontology from text"""
        try:
            schema_prompt = """Based on this text, suggest modifications to our knowledge graph schema.
Current base schema has:
- Entity types: Document, Chunk, Entity, Community
- Relation types: CONTAINS, PART_OF, RELATES_TO, IN_COMMUNITY

Analyze the text and suggest:
1. New entity types needed
2. New relationship types needed
3. Additional constraints

Respond in this EXACT format:
{
    "new_entity_types": {
        "Person": {
            "required_attributes": ["name", "role"],
            "optional_attributes": ["age", "description"]
        }
    },
    "new_relation_types": {
        "INTERACTS_WITH": {
            "valid_sources": ["Person"],
            "valid_targets": ["Person"],
            "properties": ["interaction_type"]
        }
    },
    "additional_constraints": [
        "Person entities must have unique names",
        "INTERACTS_WITH relationships must have a defined interaction_type"
    ]
}

Text to analyze: {text}

Remember:
1. Use proper JSON format
2. Include only NEW types and constraints
3. Ensure all fields are present
4. No trailing commas"""
            
            response = self.llm.invoke(schema_prompt.format(text=text))
            
            try:
                # Extract JSON part
                content = response.content.strip()
                # Find JSON boundaries
                start = content.find('{')
                end = content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    # Extract and clean JSON string
                    json_str = content[start:end]
                    # Remove any markdown formatting
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1]
                    
                    # Clean potential formatting issues
                    json_str = json_str.strip()
                    json_str = json_str.replace('\n', ' ')
                    json_str = json_str.replace('  ', ' ')
                    
                    # Parse JSON
                    modifications = json.loads(json_str)
                    
                    # Merge with base schema
                    merged_schema = dict(self.base_schema)
                    
                    if "new_entity_types" in modifications:
                        merged_schema["entity_types"].update(modifications["new_entity_types"])
                        logger.info(f"Added {len(modifications['new_entity_types'])} new entity types")
                    
                    if "new_relation_types" in modifications:
                        merged_schema["relation_types"].update(modifications["new_relation_types"])
                        logger.info(f"Added {len(modifications['new_relation_types'])} new relation types")
                    
                    if "additional_constraints" in modifications:
                        merged_schema["constraints"].extend(modifications["additional_constraints"])
                        logger.info(f"Added {len(modifications['additional_constraints'])} new constraints")
                    
                    return {
                        "schema": merged_schema,
                        "message": "Successfully merged schema modifications"
                    }
                    
                else:
                    logger.warning("No valid JSON found in response, using base schema")
                    return {
                        "schema": self.base_schema,
                        "message": "Using base schema (no valid modifications found)"
                    }
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing schema modifications: {str(e)}")
                logger.debug(f"Problematic JSON string: {json_str}")
                return {
                    "schema": self.base_schema,
                    "message": f"Using base schema due to parsing error: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error extracting ontology: {str(e)}")
            return {
                "schema": self.base_schema,
                "message": f"Using base schema due to error: {str(e)}"
            }

    def _improve_relations(self, state: GraphBuilderState) -> Dict[str, Any]:
        """Improve relationship types based on schema"""
        try:
            schema = state["schema"]
            graph = state["graph"]
            
            # For each edge, validate and improve relation type
            for u, v, data in list(graph.edges(data=True)):
                current_type = data.get('type', '')
                
                # Get entity types of connected nodes
                u_type = graph.nodes[u].get('type', '')
                v_type = graph.nodes[v].get('type', '')
                
                # Check valid relations from schema
                valid_relations = schema.get('valid_relations', {})
                type_pair = (u_type, v_type)
                
                if type_pair in valid_relations:
                    allowed_relations = valid_relations[type_pair]
                    if current_type not in allowed_relations:
                        # Ask LLM to select best valid relation
                        prompt = f"""Select the most appropriate relationship type between:
From: {u} ({u_type})
To: {v} ({v_type})
Current Relation: {current_type}
Valid Relations: {allowed_relations}

Consider the semantic meaning and context."""
                        
                        response = self.llm.invoke(prompt)
                        new_type = response.content.strip()
                        
                        if new_type in allowed_relations:
                            graph.edges[u, v]['type'] = new_type
                            logger.info(f"Improved relation: {u}-[{new_type}]->{v}")
            
            return {
                "graph": graph,
                "message": "Improved relationship types based on schema"
            }
            
        except Exception as e:
            logger.error(f"Error improving relations: {str(e)}")
            return {"graph": state["graph"], "message": str(e)}

    def _enhance_context(self, state: GraphBuilderState) -> Dict[str, Any]:
        """Enhance contextual information based on schema requirements"""
        try:
            schema = state["schema"]
            graph = state["graph"]
            
            required_context = schema.get('required_context', {})
            
            # For each node, ensure required context exists
            for node, data in graph.nodes(data=True):
                node_type = data.get('type', '')
                if node_type in required_context:
                    needed_attrs = required_context[node_type]
                    missing_attrs = [attr for attr in needed_attrs if attr not in data]
                    
                    if missing_attrs:
                        # Extract missing context from text
                        prompt = f"""Extract these contextual attributes for entity:
Entity: {node} ({node_type})
Missing Attributes: {missing_attrs}
Text Context: {state['text']}

Provide attributes in key-value format."""
                        
                        response = self.llm.invoke(prompt)
                        new_attrs = eval(response.content)
                        
                        # Add new attributes to node
                        graph.nodes[node].update(new_attrs)
                        logger.info(f"Enhanced context for {node}: {new_attrs}")
            
            return {
                "graph": graph,
                "message": "Enhanced contextual information"
            }
            
        except Exception as e:
            logger.error(f"Error enhancing context: {str(e)}")
            return {"graph": state["graph"], "message": str(e)}

    def _build_graph(self) -> StateGraph:
        """Build the graph improvement workflow"""
        workflow = StateGraph(GraphBuilderState)
        
        # Define nodes
        def analyze_node(state: GraphBuilderState) -> Dict[str, Any]:
            """Analyze current schema and metrics"""
            metrics = self._analyze_schema(state)
            return {
                "metrics": OntologyMetrics(**metrics),
                "iteration": state["iteration"] + 1
            }

        def extract_node(state: GraphBuilderState) -> Dict[str, Any]:
            """Extract ontology from text"""
            result = self._extract_ontology(state["text"])
            return {
                "schema": result["schema"]
            }

        def improve_node(state: GraphBuilderState) -> Dict[str, Any]:
            """Improve relations based on schema"""
            result = self._improve_relations(state)
            return {
                "graph": result["graph"]
            }

        def enhance_node(state: GraphBuilderState) -> Dict[str, Any]:
            """Enhance context based on schema"""
            result = self._enhance_context(state)
            return {
                "graph": result["graph"]
            }

        def end_node(state: GraphBuilderState) -> Dict[str, Any]:
            """Final node that returns the completed state"""
            return {
                "graph": state["graph"],
                "metrics": state["metrics"],
                "schema": state["schema"],
                "iteration": state["iteration"]
            }
        
        # Add nodes
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("extract", extract_node)
        workflow.add_node("improve", improve_node)
        workflow.add_node("enhance", enhance_node)
        workflow.add_node("end", end_node)
        
        # Add edges
        def should_continue(state: GraphBuilderState) -> str:
            """Determine next step based on metrics and iteration"""
            metrics = state["metrics"]
            iteration = state["iteration"]
            
            if iteration >= 5:  # Max iterations
                return "end"
            
            if metrics.average_score() > 0.8:  # Quality threshold
                return "end"
                
            # Route to appropriate improvement step
            if metrics.schema_coverage < 0.7:
                return "extract"
            elif metrics.relation_precision < 0.7:
                return "improve"
            elif metrics.context_completeness < 0.7:
                return "enhance"
            else:
                return "analyze"
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze",
            should_continue,
            {
                "extract": "extract",
                "improve": "improve",
                "enhance": "enhance",
                "end": "end"
            }
        )
        
        # Add edges back to analyze
        workflow.add_edge("extract", "analyze")
        workflow.add_edge("improve", "analyze")
        workflow.add_edge("enhance", "analyze")
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        return workflow.compile()

    def build_quality_graph(self, text: str) -> nx.MultiDiGraph:
        """Build and iteratively improve knowledge graph"""
        try:
            # Initial graph construction
            builder = KnowledgeGraphBuilder(api_key=os.getenv('GROQ_API_KEY'))
            initial_graph = builder.build(text)
            
            # Apply Chain of Thought improvements
            cot = GraphCoT(initial_graph, self.llm)
            improved_graph, walks = cot.improve_graph(num_walks=5)
            
            # Log improvement insights
            logger.info("Chain of Thought Analysis:")
            for i, walk in enumerate(walks, 1):
                logger.info(f"\nRandom Walk {i}:")
                logger.info(f"Start: {walk.starting_node}")
                logger.info(f"Path Length: {len(walk.steps)}")
                logger.info(f"Conclusion: {walk.conclusion}")
            
            return improved_graph
            
        except Exception as e:
            logger.error(f"Error building quality graph: {str(e)}")
            raise