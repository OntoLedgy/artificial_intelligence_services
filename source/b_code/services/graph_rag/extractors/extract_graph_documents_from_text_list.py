from typing import List

from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function

from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from services.graph_rag.extractors.extract_graph_documents_from_text import extract_graph_documents_from_text


@run_and_log_function()
def extract_graph_documents_from_text_list(
        texts: list,
        llm_graph_transformer: LLMGraphTransformer,
        chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING) \
        -> List[GraphDocument]:
    
    graph_documents: List[GraphDocument] = []
    
    for text in texts:
        graph_documents += extract_graph_documents_from_text(
                text,
                llm_graph_transformer,
                chunk_size)
    
    return graph_documents