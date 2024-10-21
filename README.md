# Generative AI Wrapper Services

This repository contains a set of Python wrapper services built around generative AI tools. The wrappers aim to simplify the integration and use of advanced AI functionalities in various applications, particularly for data analysis, natural language understanding, and generation.

## Tools and Services Included

### 1. **GraphRAG (Graph-based Retrieval-Augmented Generation)**
   - A wrapper service built on top of GraphRAG that helps in building knowledge graphs and integrates with generative AI for efficient query answering.
   - Enables dynamic graph construction and retrieval of information using both structured (graph) and unstructured (natural language) data.
   - Use cases:
     - Query answering with knowledge graphs.
     - Enhancing search functionalities in custom applications.

### 2. **AI Agents**
   - This module allows the creation of intelligent agents using generative AI that can autonomously interact with data, perform tasks, and make decisions.
   - Key features:
     - Customizable agents tailored for different tasks.
     - Integration with natural language models to perform operations like data extraction, summarization, and more.
     - Agents can be extended to interact with APIs, databases, or other external resources.
   - Use cases:
     - Data processing and summarization.
     - Autonomous customer support bots.
     - Task automation using AI-driven decision-making.

### 3. **LangChain Wrappers**
   - A wrapper around the popular [LangChain](https://github.com/hwchase17/langchain) library, facilitating seamless integration with various AI models for data analysis and manipulation.
   - Key features:
     - Simplifies chaining large language model (LLM) interactions with custom logic.
     - Allows creation of pipelines for data transformation, query answering, and more.
     - Supports document and knowledge-based processing.
   - Use cases:
     - Chaining multiple generative AI calls for complex workflows.
     - Integrating AI-driven insights into data pipelines.
     - Conversational agents for advanced data querying and retrieval.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- Dependencies:
  - `langchain`
  - `openai`
  - `networkx` (for GraphRAG)
  - `pandas`, `numpy` (for data handling)

## Usage

### GraphRAG Example

```python
from services.graph_rag.orchestrators.knowledge_graph_rag_from_csv_orchestrator import GraphRagOrchestrator
import pandas as pd

news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
        )

graph_rag_orchestrator = GraphRagOrchestrator(
        news)

graph_rag_orchestrator.orchestrate_graph_rag_from_csv()

```

### LangChain Wrapper Example

```python
import os
from services.summarisation.pdf_summariser import PDFSummarizer

pdf_path = r"\Kuper and Vardi - 1993 - The logical data model.pdf"
openai_api_key = os.getenv('OPENAI_API_KEY')

summarizer = PDFSummarizer(
    pdf_path,
    openai_api_key)

summarizer.load_and_split_pdf()
summary = summarizer.summarize()

print("Summary of the PDF:")
print(summary)

```
### Agents Example

```python
import asyncio
from services.agents.objects.gpt_researcher_agents import get_research_response

research_question_file_path = "../tests/data/inputs/research_question.prompt"
output_file_path = "../tests/data/outputs/test_report1.txt"

asyncio.run(
        get_research_response(
                research_question_file_path,
                output_file_path))
```

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you encounter any bugs or have suggestions for new features.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

