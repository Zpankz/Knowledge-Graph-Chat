from knowledge_graph import KnowledgeGraphBuilder
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize builder
        builder = KnowledgeGraphBuilder(api_key=api_key)

        # Read input text
        with open('test.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # Build knowledge graph
        graph = builder.build(text)
        logger.info(f"Built graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")

        # Visualize and save
        builder.visualize(save_path="knowledge_graph.png")
        logger.info("Process completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 