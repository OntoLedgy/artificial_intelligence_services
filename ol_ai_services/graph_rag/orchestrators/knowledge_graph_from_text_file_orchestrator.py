import os.path

from networkx.classes import DiGraph

from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function
from configurations.ol_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.constants import PDF_FILE_EXTENSION
from graph_rag.structure_transformers.graph_transformer_getter import get_llm_graph_transformer
from ol_ai_services.llms.client_factory import LlmClientType
from text_extraction.pdf_document_extractor import extract_text_from_pdf
from graph_rag.extractors.extract_graph_documents_from_dataset import extract_graph_documents_from_text
from data_export.networkx_digraph_from_graph_documents_getter import get_networkx_digraph_from_graph_documents


@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_text_file(
        text_file_path: str,
        model_name:str =NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_3_5_TURBO,
        temperature: float = NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE,
        client_type: LlmClientType = LlmClientType.LANGCHAIN_OPENAI) \
        -> DiGraph:
    """
    Orchestrate the process of extracting a knowledge graph from a text file.
    
    This function coordinates the end-to-end process of converting a text file into a knowledge graph:
    1. Extracts text from the provided file (currently supports PDF).
    2. Creates a graph transformer using the specified LLM.
    3. Uses the transformer to extract graph documents from the text.
    4. Converts the graph documents into a NetworkX directed graph.
    
    Args:
        text_file_path: Path to the text file (currently supports PDF format)
        model_name: The name of the language model to use (default: GPT-3.5-Turbo)
        temperature: The temperature parameter for the LLM (default: Graph RAG specific temperature)
        client_type: The type of LLM client to use, allowing selection between different providers 
                    such as OpenAI or Ollama (default: LangChain OpenAI)
    
    Returns:
        A NetworkX DiGraph representing the knowledge graph extracted from the text file
        
    Note:
        Currently only PDF text extraction is fully implemented.
    """
    
    
    # TODO: maybe use the Texts class created in the huggingface restructuring branch? - DONE
    texts = \
        __get_text_from_file(
            text_path=text_file_path)
    
    llm_graph_transformer = \
        get_llm_graph_transformer(
            model_name=model_name,
            temperature=temperature,
            client_type=client_type)
    
    graph_documents = \
        extract_graph_documents_from_text(
            text=texts,
            model_name=model_name,
            temperature=temperature,
            client_type=client_type
                )
    
    graph_document_digraph = \
        get_networkx_digraph_from_graph_documents(
                graph_documents=graph_documents)
    
    return \
        graph_document_digraph


@run_and_log_function()
def __get_text_from_file(
        text_path: str) \
        -> str:
    text_file_path_without_extension, text_file_extension = \
        os.path.splitext(
                text_path)
    
    # TODO: Only PDF texts implemented so far. DOCX needs to be added, and maybe more.
    if text_file_extension == PDF_FILE_EXTENSION:
        text = \
            extract_text_from_pdf(
                    pdf_path=text_path)
    
    else:
        text = \
            str()
    
    return \
        text



