import os.path
from langchain_experimental.graph_transformers import LLMGraphTransformer
from networkx.classes import DiGraph
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function

from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.constants import PDF_FILE_EXTENSION
from services.text_extraction.text_from_pdf_document_extractor import extract_text_from_pdf
from services.llms.chat_open_ai_session_getter import get_chat_open_ai_session
from services.graph_rag.graph_documents_from_text_extractor import extract_graph_documents_from_text
from services.data_export.networkx_digraph_from_graph_documents_getter import get_networkx_digraph_from_graph_documents


@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_text_file(
        text_file_path: str,
        model_name: str,
        temperature: float) \
        -> DiGraph:
    # TODO: maybe use the Texts class created in the huggingface restructuring branch?
    text = \
        __get_text_from_file(
            text_path=text_file_path)
    
    llm_graph_transformer = \
        __get_llm_graph_transformer(
            model_name=model_name,
            temperature=temperature)
    
    graph_documents = \
        extract_graph_documents_from_text(
            text=text,
            llm_graph_transformer=llm_graph_transformer)
    
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


@run_and_log_function()
def __get_llm_graph_transformer(
        model_name: str,
        temperature: float) \
        -> LLMGraphTransformer:
    chat_open_ai_session = \
        get_chat_open_ai_session(
                api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY,
                temperature=temperature,
                model_name=model_name)
    
    llm_graph_transformer = \
        LLMGraphTransformer(
                llm=chat_open_ai_session)
    
    return \
        llm_graph_transformer
