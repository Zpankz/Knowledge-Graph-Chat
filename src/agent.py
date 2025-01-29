"""Interactive agent for document analysis using LangGraph"""
from typing import Annotated, TypedDict, List, Dict, Any
import os
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    BaseMessage, 
    FunctionMessage, 
    SystemMessage
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
import logging

logger = logging.getLogger(__name__)


from src.graph_query import GraphQuery
from config import config
from src.graph_rag import GraphRAG

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[List[BaseMessage], add_messages]
    graph_query: GraphQuery
    context: List[Dict[str, Any]]  # Store retrieved context from GraphRAG

class DocumentAnalysisAgent:
    """Interactive agent for document analysis using graph querying"""
    
    def __init__(self, graph_query: GraphQuery):
        self.graph_query = graph_query
        
        # System prompt to guide the agent's behavior
        system_message = SystemMessage(content="""You are a helpful document analysis assistant. Your role is to:
1. Analyze the provided context from the knowledge graph
2. Use additional tools when needed to explore specific aspects
3. Provide clear, direct answers based on the document content
4. If you don't find relevant information, say so clearly
5. Don't make up information - only use what's in the document

Remember to:
- Focus on the context provided from the knowledge graph
- Use tools for additional exploration when needed
- Give specific, focused answers
- Acknowledge when information isn't available
""")
        
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name="gpt-4o",
            temperature=0
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="explore_graph",
                description="Explore specific aspects of the knowledge graph in detail",
                func=self._explore_graph,
            ),
            Tool(
                name="summarize",
                description="Summarize key points from information",
                func=self._summarize_info,
            )
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize graph components
        self.graph = self._build_graph()
        
        # Add GraphRAG
        self.graph_rag = GraphRAG(graph_query)

    def _rag_node(self, state: AgentState) -> dict:
        """GraphRAG node that retrieves relevant context"""
        query = state["messages"][-1].content
        try:
            # Get relevant documents using graph query
            documents = self.graph_query.get_relevant_documents(query)
            
            # Extract actual text content from chunks
            context = []
            for doc in documents[:1]:  # Only use the most relevant document
                # Get text content from chunk nodes
                chunk_text = None
                for node_id, node_data in self.graph_query.graph.nodes(data=True):
                    if node_data.get('type') == 'Chunk':
                        text = node_data.get('text', '')
                        if text:
                            chunk_text = text
                            break  # Only get the first chunk text
                
                if chunk_text:
                    content = chunk_text[:500]  # Limit to 500 characters
                else:
                    content = str(doc.page_content)[:500]  # Limit to 500 characters
                
                context.append({
                    "content": content,
                    "metadata": {
                        "community_id": doc.metadata.get("community_id"),
                        "num_nodes": doc.metadata.get("num_nodes", 0)
                    }
                })
            
            # Create a very concise context message
            context_message = SystemMessage(
                content=f"""Content: {context[0]['content'] if context else 'No content found'}
Analyze this content and provide a clear, concise summary."""
            )
            
            return {
                "messages": [context_message],
                "context": context
            }
            
        except Exception as e:
            error_message = f"Error retrieving context: {str(e)}"
            return {
                "messages": [SystemMessage(content=error_message)],
                "context": []
            }

    def _explore_graph(self, query: str) -> str:
        """Tool for exploring specific aspects of the knowledge graph"""
        try:
            documents = self.graph_query.get_relevant_documents(query)
            if not documents:
                return "No relevant information found."
            
            # Only use the most relevant document
            doc = documents[0]
            
            # Try to get actual text content from nodes
            for node_id, node_data in self.graph_query.graph.nodes(data=True):
                if node_data.get('type') == 'Chunk':
                    text = node_data.get('text', '')
                    if text:
                        return text[:500]  # Return first 500 characters of first found text
            
            # Fallback to document content
            return str(doc.page_content)[:500]  # Limit to 500 characters
            
        except Exception as e:
            return f"Error exploring graph: {str(e)}"

    def _summarize_info(self, text: str) -> str:
        """Tool for summarizing information"""
        try:
            # Limit input text length
            text = text[:500]  # Only summarize first 500 characters
            prompt = f"Please provide a very brief summary of this text:\n\n{text}"
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error summarizing: {str(e)}"

    def _build_graph(self) -> StateGraph:
        """Build the agent interaction graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        def agent_node(state: AgentState) -> dict:
            """Main agent node for processing queries"""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def end_node(state: AgentState) -> dict:
            """Final node that returns the last message"""
            return {"messages": state["messages"][-1:]}
        
        # Add nodes
        workflow.add_node("rag", self._rag_node)      # GraphRAG node
        workflow.add_node("agent", agent_node)        # Agent node
        workflow.add_node("tools", ToolNode(tools=self.tools))  # Tools node
        workflow.add_node("end", end_node)            # End node
        
        # Add edges
        def should_continue(state: AgentState) -> str:
            """Determine next node based on state"""
            last_message = state["messages"][-1]
            
            # If we've processed too many messages, end
            if len(state["messages"]) > 10:
                return "end"
            
            # If it's an AI message with tool calls, go to tools
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            
            # If it's a function message, continue to agent
            if isinstance(last_message, FunctionMessage):
                return "agent"
            
            # If it's an AI message without tool calls, we're done
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                return "end"
            
            # Continue with agent by default
            return "agent"
        
        # Define graph flow
        workflow.add_edge("rag", "agent")  # RAG always goes to agent first
        
        # Add conditional edges from agent node
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "agent": "agent",
                "end": "end"
            }
        )
        
        # Tools always go back to agent
        workflow.add_edge("tools", "agent")
        
        # Set entry point to RAG
        workflow.set_entry_point("rag")
        
        return workflow.compile()

    def chat(self, query: str) -> List[str]:
        """Process a user query and return responses"""
        try:
            # Get relevant documents with narrative context
            documents = self.graph_query.get_relevant_documents(query)
            
            # Extract narrative elements
            narrative_elements = {
                'text_content': [],
                'entities': set(),
                'relationships': set(),
                'communities': set()
            }
            
            for doc in documents:
                content = doc.page_content
                metadata = doc.metadata
                
                # Parse walk information
                if "Walk Analysis:" in content:
                    walk_section = content.split("Walk Analysis:")[1].split("Path Taken:")[0]
                    narrative_elements['text_content'].append(walk_section.strip())
                
                # Parse path information
                if "Path Taken:" in content:
                    path_section = content.split("Path Taken:")[1]
                    path_steps = []
                    
                    # Extract steps from content directly
                    for line in path_section.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('---'):
                            path_steps.append(line)
                    
                    narrative_elements['text_content'].extend(path_steps)
                
                # Extract metadata
                if metadata:
                    if 'start_node' in metadata:
                        narrative_elements['entities'].add(metadata['start_node'])
                    if 'path_length' in metadata:
                        narrative_elements['relationships'].add(
                            f"Path of length {metadata['path_length']}"
                        )
                    if 'conclusion' in metadata:
                        narrative_elements['text_content'].append(metadata['conclusion'])
            
            # Create narrative-focused prompt
            prompt = f"""Analyze this narrative content to answer the query.

Query: {query}

Narrative Content:
{' '.join(narrative_elements['text_content'])}

Key Elements:
{chr(10).join(f'- {entity}' for entity in narrative_elements['entities'])}

Relationships Found:
{chr(10).join(f'- {rel}' for rel in narrative_elements['relationships'])}

Provide a response that:
1. Directly answers the query
2. Uses specific details from the graph analysis
3. Explains how different elements connect
4. Maintains narrative coherence
5. Cites specific observations where relevant"""

            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Format response with source information
            final_response = [
                response.content,
                "\nSource Information:",
                f"Graph walks analyzed: {len(documents)}",
                f"Observations used: {len(narrative_elements['text_content'])}",
                f"Key elements identified: {len(narrative_elements['entities'])}",
                f"Relationships analyzed: {len(narrative_elements['relationships'])}"
            ]
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return [
                f"I encountered an error while processing your query: {str(e)}",
                "Please try rephrasing your question or providing more context."
            ]

    def run_interactive(self):
        """Run an interactive chat session"""
        print("Welcome! I'm your document analysis assistant. Type 'quit' to exit.")
        print("I can help you understand documents using the knowledge graph.")
        print("What would you like to know?\n")
        
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                responses = self.chat(query)
                for response in responses:
                    print(f"\nAssistant: {response}")
                    
            except Exception as e:
                logger.error(f"Error in interactive session: {str(e)}")
                print("\nAssistant: I apologize, but I encountered an error.")
                print("Please try asking your question in a different way.") 