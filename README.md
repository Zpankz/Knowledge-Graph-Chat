# Literary Knowledge Graph RAG

A sophisticated system that builds narrative-aware knowledge graphs from literary texts and enables natural language querying using Graph RAG (Retrieval Augmented Generation) with Chain of Thought reasoning.

## Theory & Background

This system implements several key concepts:

- **Graph-based RAG**: Extends traditional RAG by structuring information in a knowledge graph, preserving relationships and context
- **Chain of Thought (CoT) Reasoning**: Uses guided walks through the graph to build coherent narrative understanding
- **Community Detection**: Identifies thematically related subgraphs to improve retrieval relevance
- **Narrative Coherence**: Maintains story flow and context through specialized graph traversal

## Core Features

- **Intelligent Graph Construction**
  - Automatic entity and relationship extraction
  - Narrative-aware community detection
  - Dynamic graph improvement through CoT analysis

- **Advanced Query Processing**
  - Graph-guided retrieval
  - Multi-hop reasoning
  - Context-preserving response generation
  - Narrative coherence maintenance

- **Interactive Features**
  - Natural language chat interface
  - Graph visualization
  - Detailed analysis traces
  - Source attribution

## Components

### Core Modules
- `knowledge_graph.py`: Graph construction and management
- `graph_rag.py`: Graph-based retrieval system
- `graph_cot.py`: Chain of Thought reasoning engine
- `graph_query.py`: Query processing and document retrieval
- `agent.py`: Interactive chat agent

### Support Modules
- `graph_builder_agent.py`: Graph improvement agent
- `rate_limiter.py`: API rate limiting
- `models.py`: Data models and types
- `llm.py`: LLM and embedding configurations