"""Enhanced GraphRAG implementation with narrative understanding"""
from typing import Dict, List, Any
from dataclasses import dataclass
import logging
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import numpy as np
import json

from src.graph_query import GraphQuery
from config import config
import os

logger = logging.getLogger(__name__)

@dataclass
class GraphRAGResponse:
    """Response from GraphRAG including narrative context"""
    response: str
    context_data: Dict[str, Any]
    narrative_elements: Dict[str, Any]
    llm_calls: int
    prompt_tokens: int
    output_tokens: int

class GraphRAG:
    """Enhanced GraphRAG implementation with narrative focus"""
    
    def __init__(self, graph_query: GraphQuery):
        self.graph_query = graph_query
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name=config.LLM.model_name,
            temperature=config.LLM.temperature
        )
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def _get_narrative_context(self, query: str) -> List[Dict[str, Any]]:
        """Get narrative-focused context from the graph"""
        documents = self.graph_query.get_relevant_documents(query)
        
        narrative_contexts = []
        for doc in documents:
            # Extract community content
            community_content = []
            community_id = doc.metadata.get('community_id')
            
            # Get nodes in this narrative community
            community_nodes = [
                node for node, data in self.graph_query.graph.nodes(data=True)
                if data.get('type') == 'Chunk' and 
                self.graph_query.communities.get(node) == community_id
            ]
            
            # Get text and narrative elements
            narrative_elements = {
                'characters': set(),
                'events': set(),
                'settings': set(),
                'themes': set()
            }
            
            for node in community_nodes:
                node_data = self.graph_query.graph.nodes[node]
                
                # Get text content
                text = node_data.get('text', '')
                if text:
                    community_content.append(text)
                
                # Extract narrative elements
                if node_data.get('type') == 'Entity':
                    entity_type = node_data.get('entity_type', '')
                    if entity_type == 'person':
                        narrative_elements['characters'].add(node_data.get('label', ''))
                    elif entity_type == 'event':
                        narrative_elements['events'].add(node_data.get('label', ''))
                    elif entity_type == 'location':
                        narrative_elements['settings'].add(node_data.get('label', ''))
                
                # Extract themes from relationships
                for _, _, edge_data in self.graph_query.graph.edges(node, data=True):
                    if edge_data.get('type') == 'SHARED_THEME':
                        narrative_elements['themes'].add(edge_data.get('theme', ''))
            
            narrative_contexts.append({
                'content': '\n'.join(community_content),
                'elements': narrative_elements,
                'metadata': {
                    'community_id': community_id,
                    'num_nodes': len(community_nodes),
                    'title': f"Community {community_id} Narrative"
                }
            })
            
        return narrative_contexts

    def _generate_narrative_responses(self, contexts: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """Generate narrative-focused intermediate responses"""
        responses = []
        
        narrative_prompt = """Analyze this narrative context in relation to the query.
Consider characters, events, settings, and themes.

Query: {query}

Narrative Context:
{context}

Elements Present:
Characters: {characters}
Events: {events}
Settings: {settings}
Themes: {themes}

Provide analysis in this format:
{
    "key_points": [
        {
            "point": "narrative observation",
            "importance": 1-10,
            "element_type": "character/event/setting/theme",
            "evidence": "specific text evidence"
        }
    ],
    "narrative_connections": [
        {
            "elements": ["element1", "element2"],
            "relationship": "how they connect",
            "significance": "why it matters"
        }
    ]
}"""

        for context in contexts:
            try:
                elements = context['elements']
                prompt = narrative_prompt.format(
                    query=query,
                    context=context['content'][:1000],
                    characters=', '.join(elements['characters']),
                    events=', '.join(elements['events']),
                    settings=', '.join(elements['settings']),
                    themes=', '.join(elements['themes'])
                )
                
                response = self.llm.invoke(prompt)
                analysis = json.loads(response.content)
                
                responses.append({
                    'analysis': analysis,
                    'metadata': context['metadata']
                })
                
            except Exception as e:
                logger.error(f"Error generating narrative response: {str(e)}")
                continue
                
        return responses

    def _aggregate_narrative(self, responses: List[Dict[str, Any]], 
                           query: str) -> str:
        """Aggregate narrative responses into coherent story"""
        # Collect all narrative elements
        all_elements = {
            'key_points': [],
            'connections': [],
            'themes': set(),
            'characters': set()
        }
        
        for response in responses:
            analysis = response['analysis']
            
            # Collect key points
            all_elements['key_points'].extend(analysis['key_points'])
            
            # Collect narrative connections
            all_elements['connections'].extend(analysis['narrative_connections'])
            
            # Extract themes and characters
            for point in analysis['key_points']:
                if point['element_type'] == 'theme':
                    all_elements['themes'].add(point['point'])
                elif point['element_type'] == 'character':
                    all_elements['characters'].add(point['point'])
        
        # Sort points by importance
        all_elements['key_points'].sort(key=lambda x: x['importance'], reverse=True)
        
        # Generate coherent narrative
        narrative_prompt = """Create a coherent narrative that answers the query.

Query: {query}

Key Points (in order of importance):
{points}

Character Relationships:
{connections}

Major Themes: {themes}
Key Characters: {characters}

Craft a response that:
1. Directly answers the query
2. Weaves the key points into a coherent narrative
3. Shows how characters and themes develop
4. Uses specific evidence from the text
5. Acknowledges any gaps in the narrative"""

        try:
            response = self.llm.invoke(narrative_prompt.format(
                query=query,
                points=json.dumps(all_elements['key_points'], indent=2),
                connections=json.dumps(all_elements['connections'], indent=2),
                themes=', '.join(all_elements['themes']),
                characters=', '.join(all_elements['characters'])
            ))
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating final narrative: {str(e)}")
            return "Error generating narrative response"

    def query(self, query: str) -> GraphRAGResponse:
        """Process a query using narrative-focused GraphRAG"""
        try:
            # Reset counters
            self.llm_calls = 0
            self.prompt_tokens = 0
            self.output_tokens = 0
            
            # Get narrative context
            contexts = self._get_narrative_context(query)
            logger.info(f"Retrieved {len(contexts)} narrative contexts")
            
            # Generate narrative responses
            responses = self._generate_narrative_responses(contexts, query)
            logger.info(f"Generated {len(responses)} narrative analyses")
            
            # Aggregate into coherent narrative
            final_response = self._aggregate_narrative(responses, query)
            
            
            # Collect narrative elements for transparency
            narrative_elements = {
                'contexts': contexts,
                'analyses': responses,
                'themes': list(set(
                    point['content'] 
                    for response in responses
                    for point in response['analysis']['key_points']
                    if point['element_type'] == 'theme'
                )),
                'characters': list(set(
                    point['content']
                    for response in responses 
                    for point in response['analysis']['key_points']
                    if point['element_type'] == 'character'
                ))
            }
            
            return GraphRAGResponse(
                response=final_response,
                context_data={'reports': contexts},
                narrative_elements=narrative_elements,
                llm_calls=self.llm_calls,
                prompt_tokens=self.prompt_tokens,
                output_tokens=self.output_tokens
            )
            
        except Exception as e:
            logger.error(f"Error in narrative GraphRAG: {str(e)}")
            raise