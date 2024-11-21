import logging
from typing import Dict
import networkx as nx
from dotenv import load_dotenv
import os
from knowledge_graph import KnowledgeGraphBuilder
from graph_rag import GraphRetriever
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphRagTester:
    def __init__(self):
        """Initialize the tester with necessary components"""
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")
            
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.3
        )
        
        logger.info("Initialized GraphRagTester")
        
    def build_knowledge_graph(self, text: str) -> nx.MultiDiGraph:
        """Build knowledge graph from input text"""
        logger.info("Building knowledge graph...")
        builder = KnowledgeGraphBuilder(api_key=self.api_key)
        graph = builder.build(text)
        
        # Convert to MultiDiGraph if not already
        if not isinstance(graph, nx.MultiDiGraph):
            multi_graph = nx.MultiDiGraph()
            multi_graph.add_nodes_from(graph.nodes(data=True))
            multi_graph.add_edges_from(graph.edges(data=True))
            graph = multi_graph
            
        logger.info(f"Built graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        return graph
        
    def setup_retriever(self, graph: nx.DiGraph) -> GraphRetriever:
        """Setup the graph retriever"""
        logger.info("Setting up graph retriever...")
        retriever = GraphRetriever(graph=graph, llm=self.llm)
        return retriever
        
    def chat_loop(self, retriever: GraphRetriever):
        """Interactive chat loop"""
        logger.info("Starting chat loop. Type 'exit' to end.")
        print("\nChat started. Ask questions about the text (type 'exit' to end):")
        
        while True:
            # Get user input
            query = input("\nYou: ").strip()
            if query.lower() == 'exit':
                break
                
            try:
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                # Format prompt
                prompt = f"""Based on the following context from the knowledge graph, answer the question.
                
                Context:
                {context}
                
                Question: {query}
                
                Answer based only on the provided context. If you can't answer from the context, say so."""
                
                # Get response
                response = self.llm.invoke(prompt)
                print("\nAssistant:", response.content)
                
                # Print relevant nodes (for debugging)
                print("\nRelevant graph nodes used:")
                for doc in docs:
                    print(f"- Node {doc.metadata['node_id']} ({doc.metadata['node_type']})")
                    
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print("\nError processing your question. Please try again.")

def main():
    try:
        # Initialize tester
        tester = GraphRagTester()
        
        # Read test file
        logger.info("Reading test.txt...")
        with open('test.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Build knowledge graph
        graph = tester.build_knowledge_graph(text)
        
        # Setup retriever
        retriever = tester.setup_retriever(graph)
        
        # Start chat loop
        tester.chat_loop(retriever)
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 