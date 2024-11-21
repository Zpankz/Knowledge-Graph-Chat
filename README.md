# Literary Knowledge Graph RAG

A system that builds knowledge graphs from literary texts and enables natural language querying using Graph RAG (Retrieval Augmented Generation).

## Features

- **Knowledge Graph Construction**: Automatically extracts entities and relationships from text
- **Graph RAG**: Uses the knowledge graph for context-aware question answering
- **Interactive Chat**: Terminal-based interface for querying the knowledge graph
- **Visualization**: NetworkX-based graph visualization
- **Streamlit UI**: (Optional) Web interface for document processing and chat

## Components

- `knowledge_graph.py`: Core knowledge graph construction
- `graph_rag.py`: Graph-based retrieval system
- `test_graph_rag.py`: Terminal interface for testing
- `app_streamlit.py`: Streamlit web interface
- `llm.py`: Embeddings model configuration
- `rate_limiter.py`: API rate limiting utilities

## Installation

1. Clone the repository: 