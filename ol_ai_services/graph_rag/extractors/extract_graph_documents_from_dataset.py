from multiprocessing import Pool
from tqdm import tqdm
from langchain.globals import get_llm_cache
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function

from configurations.ol_configurations.nf_general_configurations import NfGeneralConfigurations
from graph_rag.extractors.extract_graph_documents_from_text import extract_graph_documents_from_text
from llms.client_factory import LlmClientType
from model_management.model_types import ModelTypes


def process_row(row):

    llm_cache = get_llm_cache()
    
    if llm_cache:
        llm_cache.clear()

    graph_documents = extract_graph_documents_from_text(
            text=f"{row['title']} {row['text']}",
            client_type=LlmClientType.LANGCHAIN_OLLAMA,
            model_name=ModelTypes.OLLAMA_MODEL_LLAMA2)
    
    return graph_documents


@run_and_log_function()
def extract_knowledge_graph_from_dataset(
        data_set):
    graph_documents = []

    with (Pool(
            processes=NfGeneralConfigurations.MAXIMUM_WORKERS)
                as pool):
        
        results = list(
                tqdm(
                        pool.imap(
                                process_row,
                                [row for _, row in data_set.iterrows()]
                                ),
                        total=len(data_set)
                        )
                )

        for graph_document in results:
            graph_documents.extend(graph_document)
            print(f"Processed {len(graph_document)} documents.")

    return graph_documents
