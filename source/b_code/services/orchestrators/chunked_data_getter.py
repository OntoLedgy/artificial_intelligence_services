from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from services.data_preparation.objects.chunked_data import ChunkedData
from services.data_preparation.objects.source_texts import SourceTexts
from services.data_preparation.pdf_services import extract_text_from_pdfs
from services.data_preparation.prepare_data import prepare_data_for_training


def get_chunked_data(
    source_documents_folder_path: str,
    chunked_data_output_file_path: str = None,
    chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING) \
    -> ChunkedData:
    pdf_texts = \
        extract_text_from_pdfs(
                pdf_folder=source_documents_folder_path)
    
    source_texts = \
        SourceTexts(
                source_texts=pdf_texts)

    chunked_texts = \
        prepare_data_for_training(
                texts=source_texts.source_texts,
                chunk_size=chunk_size)
    
    chunked_data = \
        ChunkedData(
                chunked_data=chunked_texts,
                chunk_size=chunk_size)

    if chunked_data_output_file_path:
        chunked_data.export_to_jsonl(
                output_folder_path=chunked_data_output_file_path)

    return \
        chunked_data
