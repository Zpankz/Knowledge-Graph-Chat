import streamlit as st
from faust_kg_gen import FaustKGGenerator
import logging
import os
from config import get_text_processing_config
import networkx as nx
import io
import sys
from graph_rag import GraphRetriever
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize generator with cache_resource
@st.cache_resource
def get_generator():
    """Initialize and cache the FaustKGGenerator"""
    try:
        generator = FaustKGGenerator()
        logger.info("FaustKGGenerator initialized and cached")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

# Modified process_text function with underscore prefix for generator
@st.cache_data(show_spinner=False)
def process_text(_generator, text: str):
    """Process text asynchronously"""
    try:
        # Get text processing config
        config = get_text_processing_config()
        if config["enabled"]:
            text = text[config["start_idx"]:config["end_idx"]]
            st.info(f"Testing mode: processing text from index {config['start_idx']} to {config['end_idx']}")
        
        # Process text asynchronously
        result = await _generator.process_text(text)
        
        if not result:
            raise ValueError("Processing returned no result")
            
        # Convert visualization to HTML string
        if "visualization" in result:
            viz = result["visualization"]
            temp_path = "temp_viz.html"
            viz.save_graph(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                viz_html = f.read()
            os.remove(temp_path)
            result["visualization_html"] = viz_html
            del result["visualization"]  # Remove non-serializable object
        
        # Convert graph to serializable format
        if "graph" in result:
            graph = result["graph"]
            result["graph_data"] = {
                "nodes": list(graph.nodes(data=True)),
                "edges": list(graph.edges(data=True))
            }
            del result["graph"]  # Remove non-serializable object
        
        return result
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise

class ChatState(TypedDict):
    """State for chat with graph RAG"""
    messages: Annotated[List[BaseMessage], operator.add]
    graph: nx.MultiDiGraph
    retriever: GraphRetriever

def create_graph_rag_agent(graph: nx.MultiDiGraph, llm):
    """Create a GraphRAG agent using LangGraph"""
    # Initialize retriever
    retriever = GraphRetriever(graph=graph, llm=llm)
    
    # Create graph workflow
    workflow = StateGraph(ChatState)
    
    # Define nodes
    def retrieve(state: ChatState) -> ChatState:
        """Retrieve relevant context from graph"""
        question = state["messages"][-1].content
        docs = state["retriever"].get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}
    
    def generate_response(state: ChatState) -> ChatState:
        """Generate response using context and chat history"""
        context = state.get("context", "")
        messages = state["messages"]
        
        # Format prompt with context
        system_prompt = f"""You are a helpful assistant answering questions about a text document.
        Use the following graph context to answer the question:
        
        {context}
        
        Only use information from the provided context. If you don't know, say so."""
        
        messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        # Generate response
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_response)
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()

async def process_document(text, generator, streamlit_handler, log_placeholder):
    """Process document asynchronously"""
    try:
        result = await generator.process_text(text)
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None
    finally:
        logger.removeHandler(streamlit_handler)
        if 'current_logs' in st.session_state:
            del st.session_state.current_logs

def main():
    # Set page config to wide mode
    st.set_page_config(layout="wide", page_title="Literary Knowledge Graph Chat")
    
    # Create three columns with ratio 1:1:2
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Left Column - Document Management
    with col1:
        st.title("Documents")
        
        # Initialize generator once
        generator = get_generator()
        
        # File upload
        uploaded_file = st.file_uploader("Upload Text File", type=['txt'])
        
        # Display uploaded documents
        st.subheader("Uploaded Documents")
        if uploaded_file:
            st.success(f"📄 {uploaded_file.name}")
            
            # Document info
            file_size = len(uploaded_file.getvalue()) / 1024  # Size in KB
            st.info(f"Size: {file_size:.1f} KB")
            
            # Create a placeholder for real-time logs
            log_placeholder = st.empty()
            
            # Process button
            if st.button("Process Document"):
                # Initialize session state
                if 'processed_results' not in st.session_state:
                    st.session_state.processed_results = None
                if 'log_messages' not in st.session_state:
                    st.session_state.log_messages = []
                
                # Create a custom StreamlitHandler for real-time logging
                class StreamlitHandler(logging.Handler):
                    def emit(self, record):
                        log_entry = self.format(record)
                        with log_placeholder:
                            current_logs = st.session_state.get('current_logs', [])
                            current_logs.append(log_entry)
                            st.text_area("Processing Logs", 
                                       value="\n".join(current_logs),
                                       height=400,
                                       key="log_area")
                            st.session_state.current_logs = current_logs

                # Configure the custom handler
                streamlit_handler = StreamlitHandler()
                streamlit_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                )
                logger.addHandler(streamlit_handler)
                
                # Read text
                text = uploaded_file.getvalue().decode('utf-8')
                
                # Process text
                with st.spinner("Processing..."):
                    try:
                        # Process the document asynchronously
                        async def process():
                            return await process_text(generator, text)
                        
                        result = asyncio.run(process())
                        
                        if result:
                            st.session_state.processed_results = result
                            st.success("Processing complete!")
                        else:
                            st.error("Processing failed to return results")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Processing error: {str(e)}", exc_info=True)
                    finally:
                        # Remove the handler
                        logger.removeHandler(streamlit_handler)
                        # Clear the logs from session state
                        if 'current_logs' in st.session_state:
                            del st.session_state.current_logs
    
    # Middle Column - Chat Interface
    with col2:
        st.title("Chat")
        
        if 'processed_results' in st.session_state and st.session_state.processed_results:
            # Initialize chat state if needed
            if 'chat_state' not in st.session_state:
                graph = nx.MultiDiGraph()
                # Reconstruct graph from processed results
                for node, data in st.session_state.processed_results["graph_data"]["nodes"]:
                    graph.add_node(node, **data)
                for source, target, data in st.session_state.processed_results["graph_data"]["edges"]:
                    graph.add_edge(source, target, **data)
                
                # Initialize chat agent
                st.session_state.chat_agent = create_graph_rag_agent(graph, generator.llm)
                st.session_state.chat_state = {
                    "messages": [],
                    "graph": graph,
                    "retriever": GraphRetriever(graph=graph, llm=generator.llm)
                }
            
            # Chat interface
            st.subheader("Chat with Document")
            
            # Display chat history
            for msg in st.session_state.chat_state["messages"]:
                if isinstance(msg, HumanMessage):
                    st.write("You:", msg.content)
                elif isinstance(msg, AIMessage):
                    st.write("Assistant:", msg.content)
            
            # Input for new message
            user_input = st.text_input("Your message:", key="chat_input")
            
            if user_input:
                # Add user message to state
                new_message = HumanMessage(content=user_input)
                st.session_state.chat_state["messages"].append(new_message)
                
                # Get response from agent
                with st.spinner("Thinking..."):
                    # Run the graph RAG agent
                    result = st.session_state.chat_agent.invoke(st.session_state.chat_state)
                    
                    # Update state with response
                    st.session_state.chat_state = result
                
                # Force refresh
                st.experimental_rerun()
        else:
            st.info("Please upload and process a document first.")
    
    # Right Column - Visualization
    with col3:
        st.title("Knowledge Graph")
        
        if 'processed_results' in st.session_state:
            try:
                if "visualization_html" in st.session_state.processed_results:
                    st.components.v1.html(
                        st.session_state.processed_results["visualization_html"],
                        height=800,
                        scrolling=True
                    )
                else:
                    st.info("Processing the visualization...")
                    
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                logger.error(f"Visualization error: {str(e)}", exc_info=True)
        else:
            st.info("Visualization will appear here after processing a document.")

if __name__ == "__main__":
    main()