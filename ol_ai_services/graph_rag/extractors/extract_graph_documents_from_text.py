from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function
from langchain_core.documents import Document
from configurations.ol_configurations.nf_general_configurations import NfGeneralConfigurations
from configurations.ol_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from chunking.objects.chunked_texts import ChunkedTexts
from chunking.objects.texts import Texts
from graph_rag.structure_transformers.graph_transformer_getter import get_llm_graph_transformer
from model_management.model_types import ModelTypes
from ol_ai_services.llms.client_factory import LlmClientType


@run_and_log_function()
def extract_graph_documents_from_text(
        text: str,
        model_name: str = ModelTypes.OPEN_AI_MODEL_NAME_GPT_3_5_TURBO,
        chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING,
        temperature: float = NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE,
        client_type: LlmClientType = LlmClientType.LANGCHAIN_OPENAI) \
        -> List[GraphDocument]:
    """
    Extract graph documents from a text by chunking it and processing each chunk with an LLM-based graph transformer.
    
    This function takes a text input, chunks it into manageable pieces, and then uses an LLM to extract
    entities and relationships, converting them into GraphDocument objects.
    
    Args:
        text: The source text to extract graph documents from
        model_name: The name of the language model to use for extraction (default: GPT-3.5-Turbo)
        chunk_size: The size of text chunks to process separately (default: from NfGeneralConfigurations)
        temperature: The temperature parameter for the LLM, controlling randomness in generation 
                    (default: Graph RAG specific temperature from NfOpenAiConfigurations)
        client_type: The type of LLM client to use, allowing selection between different providers like 
                    OpenAI and Ollama (default: LangChain OpenAI)
    
    Returns:
        A list of GraphDocument objects representing the entities and relationships extracted from the text
    
    Example:
        >>> text = "Alice works for Acme Corp. Bob is the CEO of Acme Corp."
        >>> graph_docs = extract_graph_documents_from_text(text)
        >>> # This would extract entities like "Alice", "Bob", "Acme Corp" and relationships like "works for", "CEO of".
    """
    
    
    llm_graph_transformer : LLMGraphTransformer= \
        get_llm_graph_transformer(
                model_name=model_name,
                temperature=temperature,
                client_type=client_type)
    
    texts = Texts(
            source_texts=[text])
    
    chunked_texts = ChunkedTexts(
            texts=texts,
            chunk_size=chunk_size,
            output_file_path=''
            )
    
    graph_documents = []
    
    for chunk_dict in chunked_texts.chunked_texts:
        
        chunk = chunk_dict['text']
        
        doc = Document(
                page_content=chunk)
        
        graph_documents.extend(
                llm_graph_transformer.convert_to_graph_documents(
                        [doc]))
    
    return graph_documents