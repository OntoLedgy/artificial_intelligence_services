# Artificial Intelligence Services

A comprehensive Python toolkit for leveraging advanced generative AI capabilities across multiple domains. This repository provides high-level abstractions, workflow orchestrators, and specialized components to simplify the integration of cutting-edge AI technologies into enterprise applications.

## Core Capabilities

### 1. **GraphRAG (Graph-based Retrieval-Augmented Generation)**
   - Build, query, and manage knowledge graphs from various data sources (PDF, text files, CSV, Word documents)
   - Integrate Neo4j for advanced graph storage, querying, and visualization
   - Extract entities and relationships from unstructured text using LLMs
   - Perform contextualized question-answering using the power of graph-based context retrieval
   - Support for multiple graph modeling approaches and visualization tools
   - Use cases:
     - Knowledge management and discovery in enterprise documents
     - Complex relationship extraction from unstructured data
     - Enhanced contextual search across document repositories
     - Building domain-specific knowledge graphs dynamically

### 2. **AI Agents and Orchestration**
   - Create autonomous AI agents for specialized tasks using LangGraph architecture
   - GPT Researcher integration for automated research and information gathering
   - Coding agent capabilities for automated code generation and analysis
   - DeepWiki integration for knowledge management
   - Agent workflow orchestration with state management
   - Multi-agent systems with specialized agent roles
   - Use cases:
     - Automated research and report generation
     - Intelligent code assistance and generation
     - Complex multi-step workflow automation
     - Self-improving agent systems with feedback loops

### 3. **Document Processing Pipeline**
   - Extract and process text from various document formats (PDF, Word, text)
   - Intelligent document chunking with customizable strategies
   - Advanced tokenization with multiple tokenizer backends (OpenAI, Hugging Face)
   - Document structure extraction and analysis
   - Batch processing capabilities for document folders
   - Use cases:
     - Document summarization and analysis
     - Information extraction from large document collections
     - Preparing documents for semantic search and RAG applications

### 4. **LLM Integration and Management**
   - Unified interface for multiple LLM providers (OpenAI, potentially others)
   - Model fine-tuning capabilities with training pipeline orchestration
   - Text generation with configurable parameters and constraints
   - LangChain integration for complex chains and workflows
   - Model management and versioning
   - Use cases:
     - Domain-specific model customization
     - Controlled text generation for various applications
     - Building complex LLM-powered workflows

### 5. **Embedding and Semantic Search**
   - Generate and manage text embeddings for semantic understanding
   - Search embedded documents by similarity
   - Integration with vector stores for efficient retrieval
   - Support for various embedding models and strategies
   - Use cases:
     - Semantic search across document repositories
     - Similar document discovery
     - Building embedding-based recommendation systems

## Installation and Setup

### Using Poetry (Recommended)

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies using Poetry
poetry install
```

### Using pip

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- Core Dependencies:
  - `langchain` - Framework for LLM application development
  - `openai` - OpenAI API client
  - `networkx` - Graph data structure and algorithms
  - `neo4j` - Neo4j graph database client
  - `pandas`, `numpy` - Data manipulation libraries
  - `PyPDF2`, `python-docx` - Document processing
  - `transformers` - Hugging Face Transformer models
  - `tiktoken` - OpenAI tokenizer
  - `faiss-cpu` - Vector similarity search
  - `langgraph` - Agent orchestration framework

## Usage Examples

### Configuration Setup

Before using the services, set up your configuration:

```python
import os
from configurations.ol_configurations.open_ai_configurations import OpenAIConfigurations

# Set your OpenAI API key in environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Or configure through the configurations module
config = OpenAIConfigurations()
config.set_api_key("your-api-key")
```

### GraphRAG Example: Building Knowledge Graph from Tabular Data

```python
from ol_ai_services.graph_rag.orchestrators.knowledge_graph_from_tabular_dataset_orchestrator import KnowledgeGraphFromTabularDatasetOrchestrator
import pandas as pd

# Load your dataset
news_data = pd.read_csv("path/to/news_articles.csv")

# Initialize the orchestrator
orchestrator = KnowledgeGraphFromTabularDatasetOrchestrator(news_data)

# Generate knowledge graph
knowledge_graph = orchestrator.orchestrate_retrieve_knowledge_graph_from_tabular_data_set()

# Query the graph
results = orchestrator.query_knowledge_graph("What are the main themes in the news articles?")
print(results)
```

### Document Processing and Summarization

```python
from ol_ai_services.summarisation.pdf_summariser import PDFSummariser
from ol_ai_services.llms.clients.open_ai_clients import OpenAIClient

# Initialize client and summarizer
client = OpenAIClient()
summarizer = PDFSummariser(
    pdf_file_path="path/to/document.pdf", 
    open_ai_client=client
)

# Generate summary
summary = summarizer.summarise_pdf()
print(f"Document Summary:\n{summary}")
```

### AI Research Agent Example

```python
import asyncio
from ol_ai_services.agents.objects.gpt_researcher_agents import GPTResearcherAgent

# Define research question
research_question = "What are the latest advancements in quantum computing?"
output_file_path = "research_report.txt"

# Initialize and run research agent
researcher = GPTResearcherAgent()
asyncio.run(researcher.conduct_research(research_question, output_file_path))

print(f"Research completed and saved to {output_file_path}")
```

### Coding Agent Example

```python
from ol_ai_services.agents.coding.coding_agent import CodingAgent

# Initialize coding agent
coding_agent = CodingAgent()

# Generate code based on requirements
code = coding_agent.generate_code(
    "Create a Python function that calculates the Fibonacci sequence up to n terms"
)

# Execute and validate code
result = coding_agent.execute_code(code)
print(f"Code execution result:\n{result}")
```

### Text Embedding and Semantic Search

```python
from ol_ai_services.embeddings.search_embedded_documents import SearchEmbeddedDocuments
from ol_ai_services.chunking.chunked_texts_getter import ChunkedTextsGetter

# Process and chunk documents
chunker = ChunkedTextsGetter()
chunked_texts = chunker.get_chunked_texts_from_text_file("path/to/document.txt")

# Create search engine
search_engine = SearchEmbeddedDocuments(chunked_texts)

# Perform semantic search
results = search_engine.search("quantum computing applications", top_k=3)
for result in results:
    print(f"Score: {result.score}, Content: {result.text}")
```

## Architecture and Components

### Project Structure

```
artificial_intelligence_services/
├── ol_ai_services/              # Core AI service implementations
│   ├── agents/                  # AI agent implementations
│   ├── chunking/                # Text chunking utilities
│   ├── common_utilities/        # Shared utilities
│   ├── data_export/             # Data export tools
│   ├── embeddings/              # Text embedding services
│   ├── fine_tuning/             # Model fine-tuning capabilities
│   ├── graph_rag/               # Graph-based RAG implementation
│   │   ├── extractors/          # Entity and relation extraction
│   │   ├── integrators/         # Graph database integration
│   │   ├── orchestrators/       # Workflow orchestrators
│   │   ├── resolvers/           # Entity resolution
│   │   └── validators/          # Graph validation utilities
│   ├── llms/                    # LLM integrations
│   ├── model_management/        # Model loading and management
│   ├── summarisation/           # Document summarization
│   ├── text_extraction/         # Document text extraction
│   └── tokenisation/            # Text tokenization utilities
├── configurations/              # Configuration management
└── tests/                       # Comprehensive test suite
```

### Supported Models and Integrations

- **Language Models**:
  - OpenAI GPT models (GPT-4, GPT-3.5-turbo)
  - Hugging Face Transformer models
  
- **Vector Databases**:
  - FAISS for efficient vector storage and retrieval
  
- **Graph Databases**:
  - Neo4j with comprehensive data modeling and query capabilities
  
- **Framework Integrations**:
  - LangChain for building complex LLM applications
  - LangGraph for agent-based workflows
  - Transformers for model fine-tuning and inference

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

