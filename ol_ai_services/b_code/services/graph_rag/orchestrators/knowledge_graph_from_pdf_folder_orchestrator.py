from networkx.classes import DiGraph

from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function

from services.data_export.networkx_digraph_from_graph_documents_getter import get_networkx_digraph_from_graph_documents
from services.graph_rag.extractors.extract_graph_documents_from_dataset import extract_knowledge_graph_from_dataset

from services.text_extraction.pdf_folder_extractor import extract_dataframe_from_pdfs_in_folder


@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_pdf_folder_file(
        folder_path: str) \
        -> DiGraph:
    
    pdf_documents = extract_dataframe_from_pdfs_in_folder(
            directory_path=folder_path,
            looks_into_subfolders=True,
            )
    
    knowledge_directed_graph = extract_knowledge_graph_from_dataset(
            data_set=pdf_documents
            )
    
    graph_document_digraph = \
        get_networkx_digraph_from_graph_documents(
                graph_documents=knowledge_directed_graph)
    
    return graph_document_digraph