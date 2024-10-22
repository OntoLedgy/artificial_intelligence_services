import os.path

from networkx.classes import DiGraph

from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.constants import PDF_FILE_EXTENSION
from services.graph_rag.transformers.graph_transformer_getter import get_llm_graph_transformer
from services.text_extraction.pdf_document_extractor import extract_text_from_pdf
from services.graph_rag.extractors.extract_graph_documents_from_dataset import extract_graph_documents_from_text
from services.data_export.networkx_digraph_from_graph_documents_getter import get_networkx_digraph_from_graph_documents


#@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_text_file(
        text_file_path: str,
        model_name:str =NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_3_5_TURBO,
        temperature: float = NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE) \
        -> DiGraph:
    
    # TODO: maybe use the Texts class created in the huggingface restructuring branch? - DONE
    texts = \
        __get_text_from_file(
            text_path=text_file_path)
    
    llm_graph_transformer = \
        get_llm_graph_transformer(
            model_name=model_name,
            temperature=temperature)
    
    graph_documents = \
        extract_graph_documents_from_text(
            text=texts,
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



