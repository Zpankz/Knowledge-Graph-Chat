"""Graph Chain of Thought implementation using random walks"""
from typing import List, Dict, Any, Set, Tuple
import random
import networkx as nx
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import logging
import json
from src.llm import embeddings

logger = logging.getLogger(__name__)

class ThoughtStep(BaseModel):
    """Single step in the chain of thought"""
    current_node: str = Field(..., description="Current node being analyzed")
    observation: str = Field(..., description="What is observed at this node")
    reasoning: str = Field(..., description="Reasoning about the observation")
    next_step: str = Field(..., description="Decision about next step")

class RandomWalk(BaseModel):
    """Complete random walk through the graph"""
    starting_node: str = Field(..., description="Node where walk started")
    steps: List[ThoughtStep] = Field(..., description="Steps in the walk")
    conclusion: str = Field(..., description="Insights from this walk")

class GraphCoT:
    """Chain of Thought processor for knowledge graphs"""
    
    def __init__(self, graph: nx.MultiDiGraph, llm: ChatOpenAI):
        self.graph = graph
        self.llm = llm
        
    def _select_random_node(self, exclude: Set[str] = None) -> str:
        """Select a random node, optionally excluding some nodes"""
        available_nodes = list(set(self.graph.nodes()) - (exclude or set()))
        if not available_nodes:
            raise ValueError("No available nodes to select from")
        return random.choice(available_nodes)
    
    def _get_node_context(self, node: str) -> Dict[str, Any]:
        """Get rich context for a node"""
        if node not in self.graph:
            raise ValueError(f"Node {node} not found in graph")
        node_data = self.graph.nodes[node]
        neighbors = list(self.graph.neighbors(node))
        edges = list(self.graph.edges(node, data=True))
        
        return {
            "node_id": node,
            "type": node_data.get("type", "unknown"),
            "attributes": node_data,
            "neighbors": neighbors,
            "relationships": edges
        }
    
    def _analyze_step(self, node: str, path_so_far: List[str]) -> ThoughtStep:
        """Analyze current step in the walk with narrative focus"""
        context = self._get_node_context(node)
        
        # First, get node content
        node_content = ""
        if 'text' in context['attributes']:
            node_content = context['attributes']['text']
        elif 'label' in context['attributes']:
            node_content = context['attributes']['label']
            
        # Get relationship descriptions
        relationships = []
        for _, target, data in context['relationships']:
            rel_type = data.get('type', 'RELATES_TO')
            target_data = self.graph.nodes[target]
            target_label = target_data.get('label', target)
            relationships.append(f"{rel_type} -> {target_label}")

        prompt = """Analyze this position in the narrative graph.
DO NOT include any additional text, ONLY the JSON response.

Node Information:
ID: {node_id}
Type: {type}
Content: {content}
Connected To: {connections}

Previous Path: {path}

RESPOND EXACTLY IN THIS FORMAT:
{{
    "observation": "What you directly observe about this node and its immediate connections",
    "reasoning": "Why this node and its connections are significant to understanding the narrative",
    "next_step": "Which connection would be most valuable to explore next and why"
}}"""

        try:
            # Format prompt with specific information
            formatted_prompt = prompt.format(
                node_id=context["node_id"],
                type=context["type"],
                content=node_content[:200],  # Limit content length
                connections="; ".join(relationships),
                path=" -> ".join(path_so_far)
            )
            
            response = self.llm.invoke(formatted_prompt)
            
            # Clean and parse response
            content = response.content.strip()
            
            # Handle potential markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            
            # Clean whitespace and ensure valid JSON
            content = content.strip()
            content = ' '.join(content.split())
            
            # Ensure we have a complete JSON object
            if not content.startswith('{'):
                content = '{' + content.split('{', 1)[1]
            if not content.endswith('}'):
                content = content.rsplit('}', 1)[0] + '}'
            
            # Parse and validate
            analysis = json.loads(content)
            
            # Ensure all required fields are present
            required_fields = ["observation", "reasoning", "next_step"]
            if not all(field in analysis for field in required_fields):
                missing = [f for f in required_fields if f not in analysis]
                raise ValueError(f"Missing required fields: {missing}")
            
            return ThoughtStep(
                current_node=node,
                observation=analysis["observation"],
                reasoning=analysis["reasoning"],
                next_step=analysis["next_step"]
            )
            
        except Exception as e:
            logger.error(f"Error in narrative analysis: {str(e)}")
            logger.debug(f"Problematic response: {response.content if 'response' in locals() else 'No response'}")
            
            # Return fallback analysis
            return self._fallback_analysis(node, context)

    def _fallback_analysis(self, node: str, context: Dict[str, Any]) -> ThoughtStep:
        """Provide basic narrative analysis when detailed analysis fails"""
        node_type = context['type']
        
        if node_type == 'Chunk':
            observation = f"This is a text chunk that might contain part of the story"
            reasoning = "Text chunks help us understand the narrative sequence"
            next_step = "Look for connected story elements"
        elif node_type == 'Entity':
            observation = f"This is an entity (possibly a character or story element)"
            reasoning = "Entities are key components of the narrative"
            next_step = "Explore relationships with other characters/elements"
        else:
            observation = f"This is a {node_type} node in the story graph"
            reasoning = "Every node contributes to the overall narrative"
            next_step = "Follow strongest narrative connection"
            
        return ThoughtStep(
            current_node=node,
            observation=observation,
            reasoning=reasoning,
            next_step=next_step
        )
    
    def _random_walk(self, max_depth: int = 5, exclude_starts: Set[str] = None) -> RandomWalk:
        """Perform a single random walk through the graph"""
        try:
            start_node = self._select_random_node(exclude_starts)
            current_node = start_node
            path = [start_node]
            steps = []
            
            for depth in range(max_depth):
                # Analyze current position
                step = self._analyze_step(current_node, path)
                if step:  # Only append valid steps
                    steps.append(step)
                
                # Get possible next nodes
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                    
                # Select next node
                current_node = random.choice(neighbors)
                path.append(current_node)
            
            # Generate conclusion for this walk
            conclusion_prompt = """Analyze this walk through the knowledge graph:

Starting Node: {start}
Path Taken: {path}
Key Observations: {observations}

What insights can we draw about the narrative structure and content from this walk?
Provide a clear, concise summary of the key findings."""

            observations = "\n".join([
                f"- At {step.current_node}: {step.observation}"
                for step in steps
            ])

            conclusion = self.llm.invoke(conclusion_prompt.format(
                start=start_node,
                path=" -> ".join(path),
                observations=observations
            )).content
            
            return RandomWalk(
                starting_node=start_node,
                steps=steps,
                conclusion=conclusion
            )
            
        except Exception as e:
            logger.error(f"Error in random walk: {str(e)}")
            # Return a minimal valid RandomWalk object
            return RandomWalk(
                starting_node=start_node if 'start_node' in locals() else "unknown",
                steps=[],
                conclusion="Error occurred during walk analysis"
            )
    
    def improve_graph(self, num_walks: int = 5) -> Tuple[nx.MultiDiGraph, List[RandomWalk]]:
        """Improve graph using narrative-focused random walks"""
        walks = []
        start_nodes = set()
        
        # Track narrative improvements
        narrative_improvements = {
            'new_connections': 0,
            'enhanced_context': 0,
            'strengthened_communities': 0
        }
        
        for i in range(num_walks):
            logger.info(f"\nStarting narrative walk {i+1}/{num_walks}")
            walk = self._random_walk(exclude_starts=start_nodes)
            walks.append(walk)
            
            # Analyze narrative coherence
            coherence_prompt = """Analyze the narrative coherence of this walk through the story:

Path: {path}
Observations: {observations}

Suggest improvements to strengthen the narrative in this format:
{{
    "new_connections": [
        {{
            "source": "node_id",
            "target": "node_id",
            "type": "relationship_type",
            "narrative_reason": "how this connection enhances the story"
        }}
    ],
    "context_enhancements": [
        {{
            "node": "node_id",
            "additions": {{"attribute": "value"}},
            "narrative_purpose": "how this context enriches the story"
        }}
    ],
    "community_suggestions": [
        {{
            "nodes": ["node_id1", "node_id2"],
            "theme": "shared narrative theme",
            "reason": "why these elements belong together"
        }}
    ]
}}"""

            try:
                improvements = self._get_narrative_improvements(
                    walk, coherence_prompt
                )
                
                # Apply narrative improvements
                self._apply_narrative_improvements(
                    improvements, narrative_improvements
                )
                
            except Exception as e:
                logger.error(f"Error improving narrative: {str(e)}")
                continue
                
        # Log improvement statistics
        logger.info("\nNarrative Improvement Statistics:")
        logger.info(f"New Connections: {narrative_improvements['new_connections']}")
        logger.info(f"Enhanced Context: {narrative_improvements['enhanced_context']}")
        logger.info(f"Strengthened Communities: {narrative_improvements['strengthened_communities']}")
        
        return self.graph, walks

    def _get_narrative_improvements(self, walk: RandomWalk, prompt: str) -> Dict[str, Any]:
        """Get narrative improvement suggestions from LLM"""
        observations = "\n".join([
            f"- At {step.current_node}: {step.observation}"
            for step in walk.steps
        ])
        
        response = self.llm.invoke(prompt.format(
            path=" -> ".join([step.current_node for step in walk.steps]),
            observations=observations
        ))
        
        return self._parse_llm_response(response.content)

    def _apply_narrative_improvements(self, improvements: Dict[str, Any], 
                                   stats: Dict[str, int]) -> None:
        """Apply narrative improvements to the graph"""
        # Add new narrative connections
        for conn in improvements.get("new_connections", []):
            if conn["source"] in self.graph and conn["target"] in self.graph:
                self.graph.add_edge(
                    conn["source"],
                    conn["target"],
                    type=conn["type"],
                    narrative_reason=conn["narrative_reason"]
                )
                stats['new_connections'] += 1
                logger.info(f"Added narrative connection: {conn['source']} -> {conn['target']}"
                          f" ({conn['narrative_reason']})")
        
        # Enhance narrative context
        for enh in improvements.get("context_enhancements", []):
            if enh["node"] in self.graph:
                self.graph.nodes[enh["node"]].update(enh["additions"])
                stats['enhanced_context'] += 1
                logger.info(f"Enhanced narrative context for {enh['node']}: "
                          f"{enh['narrative_purpose']}")
        
        # Strengthen narrative communities
        for comm in improvements.get("community_suggestions", []):
            valid_nodes = [n for n in comm["nodes"] if n in self.graph]
            if len(valid_nodes) > 1:
                for i in range(len(valid_nodes)-1):
                    for j in range(i+1, len(valid_nodes)):
                        self.graph.add_edge(
                            valid_nodes[i],
                            valid_nodes[j],
                            type="SHARED_THEME",
                            theme=comm["theme"],
                            reason=comm["reason"]
                        )
                stats['strengthened_communities'] += 1
                logger.info(f"Strengthened narrative community: {comm['theme']}")
    
    def analyze_for_query(self, query: str, num_walks: int = 3) -> List[Dict[str, Any]]:
        """Use random walks to find relevant information for a query"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        relevant_info = []
        
        # Embed query for similarity comparison
        query_embedding = embeddings.embed_query(query)
        
        # Perform random walks
        for i in range(num_walks):
            walk = self._random_walk()
            
            # Analyze walk for query relevance
            analyze_prompt = """Analyze this walk through the knowledge graph for relevance to the query.

Query: {query}
Walk: {walk}

How relevant is the information in this walk to answering the query?
Provide specific examples of relevant information found."""

            response = self.llm.invoke(analyze_prompt.format(
                query=query,
                walk=walk.model_dump()
            ))
            
            relevant_info.append({
                "walk": walk.model_dump(),
                "analysis": response.content
            })
        
        return relevant_info
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse and clean LLM response"""
        try:
            # Clean the response
            content = content.strip()
            
            # Remove any markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            
            # Clean whitespace and ensure valid JSON
            content = content.strip()
            content = ' '.join(content.split())
            if not content.startswith('{'):
                content = '{' + content.split('{', 1)[1]
            if not content.endswith('}'):
                content = content.rsplit('}', 1)[0] + '}'
            
            # Parse JSON
            result = json.loads(content)
            
            # Validate required fields based on context
            if "observation" in result:  # Step analysis
                required = ["observation", "reasoning", "next_step"]
                if not all(k in result for k in required):
                    raise ValueError(f"Missing required fields. Found: {list(result.keys())}")
            
            elif "new_connections" in result:  # Narrative improvements
                required = ["new_connections", "context_enhancements", "community_suggestions"]
                if not all(k in result for k in required):
                    raise ValueError(f"Missing required fields. Found: {list(result.keys())}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.debug(f"Problematic content: {content}")
            
            # Return safe default based on context
            if "observation" in content.lower():
                return {
                    "observation": "Failed to parse detailed observation",
                    "reasoning": "Analysis failed",
                    "next_step": "Continue to random neighbor"
                }
            else:
                return {
                    "new_connections": [],
                    "context_enhancements": [],
                    "community_suggestions": []
                }