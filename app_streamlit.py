"""
Streamlit interface for Knowledge Graph application with MASTER framework integration
"""
import streamlit as st
import networkx as nx
from pathlib import Path
import os
from dotenv import load_dotenv
from config import config
from main import KnowledgeGraphPipeline

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("OPENAI_API_KEY not found in environment variables")
    st.stop()

# Use config for Streamlit settings
st.set_page_config(**config.STREAMLIT_CONFIG)

st.title("Knowledge Graph Explorer 🧠")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = KnowledgeGraphPipeline(api_key)
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = Path(config.GRAPH_SAVE_PATH).exists()

# Sidebar
st.sidebar.header("Build Graph")

# Input method selection
input_method = st.sidebar.radio(
    "Choose input method",
    ["Text Input", "File Upload"]
)

if input_method == "Text Input":
    input_text = st.sidebar.text_area("Enter text to build graph from", height=300)
    if st.sidebar.button("Build Graph from Text"):
        with st.spinner("Building knowledge graph..."):
            st.session_state.pipeline.build_graph(input_text)
            st.session_state.graph_built = True
            st.success("Graph built successfully!")
            st.rerun()

else:  # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload a text file", type=['txt'])
    if uploaded_file and st.sidebar.button("Build Graph from File"):
        with st.spinner("Building knowledge graph..."):
            input_text = uploaded_file.getvalue().decode()
            st.session_state.pipeline.build_graph(input_text)
            st.session_state.graph_built = True
            st.success("Graph built successfully!")
            st.rerun()

# Main content area
if st.session_state.graph_built:
    # Load the graph
    G = nx.read_gml(config.GRAPH_SAVE_PATH)
    
    # Display basic graph statistics
    st.header("Graph Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", len(G.nodes()))
    with col2:
        st.metric("Edges", len(G.edges()))
    with col3:
        st.metric("Communities", len(set(nx.get_node_attributes(G, 'community').values())))
    
    # Query interface
    st.header("Query Graph")
    
    # Add query method selection
    query_method = st.radio(
        "Choose query method",
        ["Standard Query", "MASTER Framework"],
        help="Standard Query uses direct graph traversal. MASTER Framework uses Monte Carlo Tree Search for more thorough exploration."
    )
    
    query = st.text_input("Enter your query")
    if query:
        if query_method == "Standard Query":
            with st.spinner("Querying graph..."):
                try:
                    documents = st.session_state.pipeline.query_graph(query)
                    st.subheader("Query Results")
                    for i, doc in enumerate(documents, 1):
                        with st.expander(f"Document {i}"):
                            st.write(doc)
                except Exception as e:
                    st.error(f"Error querying graph: {str(e)}")
                    
        else:  # MASTER Framework
            from src.master import MASTER
            import graphviz
            
            # MASTER Configuration in sidebar
            st.sidebar.markdown("### MASTER Framework Settings")
            max_expansion = st.sidebar.slider("Max Expansions", 1, 5, 3,
                help="Number of expansion rounds in Monte Carlo Tree Search")
            num_branches = st.sidebar.slider("Branches per Expansion", 1, 3, 2,
                help="Number of child solutions to generate in each expansion")
            
            with st.spinner("Running MASTER framework..."):
                try:
                    # Get initial relevant documents
                    documents = st.session_state.pipeline.query_graph(query)
                    
                    with st.status("🤔 Thinking with MASTER...") as status:
                        # Run MASTER with document context
                        status.write("Initializing MASTER framework...")
                        master = MASTER(
                            question=query,
                            context=documents,
                            max_expansion=max_expansion,
                            num_branches=num_branches,
                            llm=st.session_state.pipeline.llm
                        )
                        
                        status.write("Solving with Monte Carlo Tree Search...")
                        result = master.solve()
                    
                    # Create tabs for different visualizations
                    solution_tab, tree_tab = st.tabs([
                        "Solution", "Agent Tree"
                    ])
                    
                    # Solution Tab
                    with solution_tab:
                        st.subheader("Final Answer")
                        st.write(result.solution)
                        
                        # Show solution details
                        with st.expander("Solution Analysis", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Solution Score", f"{result.score:.2f}/10")
                            with col2:
                                st.metric("Confidence", f"{result.confidence:.2%}")
                            
                            st.markdown("#### Validation")
                            st.write(result.validation)
                    
                    # Agent Tree Tab
                    with tree_tab:
                        st.subheader("Solution Search Tree")
                        graph = graphviz.Digraph()
                        
                        # Add nodes with solution information
                        for agent_id, agent in master.agents.items():
                            # Create label with core metrics
                            label = (
                                f"Agent {agent_id}\n"
                                f"Score: {agent.score:.2f}\n"
                                f"Conf: {agent.confidence:.2f}"
                            )
                            if agent.terminal:
                                label += f"\n{'✅ Pass' if agent.passed else '❌ Fail'}"
                            
                            # Color based on score
                            color = f"#{int(agent.score * 25.5):02x}ff{int((10-agent.score) * 25.5):02x}"
                            
                            graph.node(str(agent_id), label=label,
                                     style='filled', fillcolor=color)
                            
                            if agent.parent:
                                graph.edge(str(agent.parent.agent_id), str(agent_id))
                        
                        st.graphviz_chart(graph)
                        
                        # Show statistics
                        st.markdown("#### Search Statistics")
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.metric("Total Agents", len(master.agents))
                            st.metric("Terminal Solutions",
                                    len([a for a in master.agents.values() if a.terminal]))
                        with stats_col2:
                            st.metric("Average Score",
                                    f"{sum(a.score for a in master.agents.values()) / len(master.agents):.2f}")
                            st.metric("Success Rate",
                                    f"{len([a for a in master.agents.values() if a.passed])/len(master.agents):.1%}")
                    
                    # Show source documents
                    if documents:
                        with st.expander("Source Documents"):
                            for i, doc in enumerate(documents, 1):
                                st.markdown(f"**Document {i}:**")
                                st.write(doc)
                    
                except Exception as e:
                    st.error(f"Error in MASTER framework: {str(e)}")
                    st.error(f"Details: {str(e.__class__.__name__)}")
    
    # Display communities
    st.header("Communities")
    communities = {}
    for node, data in G.nodes(data=True):
        comm = data.get('community', 'Unknown')
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)
    
    for comm, nodes in sorted(communities.items()):
        with st.expander(f"Community {comm} ({len(nodes)} nodes)"):
            st.write(", ".join(nodes))
            
    # Display graph visualization if available
    if os.path.exists(config.VISUALIZATION_PATH):
        st.header("Graph Visualization")
        st.image(config.VISUALIZATION_PATH)

else:
    st.info("Start by building a knowledge graph using the sidebar options.")