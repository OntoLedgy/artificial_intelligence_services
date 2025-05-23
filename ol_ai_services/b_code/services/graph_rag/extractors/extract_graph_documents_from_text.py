from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from langchain_core.documents import Document
from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from services.chunking.objects.chunked_texts import ChunkedTexts
from services.chunking.objects.texts import Texts
from services.graph_rag.transformers.graph_transformer_getter import get_llm_graph_transformer


@run_and_log_function()
def extract_graph_documents_from_text(
        text: str,
        model_name: str = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_3_5_TURBO,
        chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING,
        temperature: float = NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE) \
        -> List[GraphDocument]:
    
    llm_graph_transformer : LLMGraphTransformer= \
        get_llm_graph_transformer(
                model_name=model_name,
                temperature=temperature)
    
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