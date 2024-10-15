from typing import List
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function


@run_and_log_function()
def extract_graph_documents_from_text(
        text: str,
        llm_graph_transformer: LLMGraphTransformer) \
        -> List[GraphDocument]:
    text_in_langchain_document_format = \
        Document(
            page_content=text)
    
    graph_documents = \
        llm_graph_transformer.convert_to_graph_documents(
            documents=[text_in_langchain_document_format])
    
    return \
        graph_documents
