from networkx.classes import DiGraph
from pandas import DataFrame
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function

from graph_rag.extractors.extract_graph_documents_from_dataset import extract_knowledge_graph_from_dataset

from text_extraction.word_document_sections_extractor import extract_text_from_word_document_sections


@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_word_file(
        word_file_path: str) \
        -> DiGraph:
    
    word_document_sections = extract_text_from_word_document_sections(
            document_file_path=word_file_path
            )
    word_document_text_list : list[str] = []
    
    for word_document_section in word_document_sections:
        word_document_text_list += word_document_section['content']
    
    word_document_data = {
        'title': [section['heading'] for section in word_document_sections],
        'text' : [section['content'] for section in word_document_sections]
        }
    
    word_documents_data = DataFrame(
        word_document_data)
    
    knowledge_directed_graph = extract_knowledge_graph_from_dataset(
        data_set= word_documents_data,
        )
    
    return knowledge_directed_graph







