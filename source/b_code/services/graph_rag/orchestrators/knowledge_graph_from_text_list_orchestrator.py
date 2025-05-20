from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function

from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from services.graph_rag.extractors.extract_graph_documents_from_text import extract_graph_documents_from_text


@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_text_list(
        text_list,
        maximum_workers = NfGeneralConfigurations.MAXIMUM_WORKERS
        ):
    
    graph_documents = []
    with ThreadPoolExecutor(
            max_workers=maximum_workers) as executor:
        
        futures = [
            executor.submit(
                    extract_graph_documents_from_text,
                    text_to_be_processed
                    )
            for text_to_be_processed in text_list
            ]
        
        for future in tqdm(
                as_completed(
                        futures),
                total=len(
                        futures),
                desc="Processing documents"):
            try:
                graph_document = future.result()
                graph_documents.extend(
                        graph_document)
                print(
                        f"Processed {len(graph_document)} documents.")
            
            except Exception as e:
                print(
                        e)
    

    return \
        graph_documents