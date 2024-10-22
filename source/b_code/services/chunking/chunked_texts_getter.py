import os

from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from configurations.constants import PDF_FILE_EXTENSION
from services.chunking.objects.chunked_texts import ChunkedTexts
from services.chunking.objects.texts import Texts
from services.text_extraction.pdf_document_extractor import extract_text_from_pdfs_in_folder


def get_chunked_texts(
        source_texts_folder_path: str,
        chunked_texts_output_file_path: str = None,
        extension: str = PDF_FILE_EXTENSION,
        chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING) \
    -> ChunkedTexts:
    # TODO: Only PDF implemented at the moment
    if extension == PDF_FILE_EXTENSION:
        source_texts = \
            extract_text_from_pdfs_in_folder(
                    directory=source_texts_folder_path)
    
    else:
        source_texts = \
            list()
        
    texts = \
        Texts(
            source_texts=source_texts,
            output_folder_path=os.path.dirname(
                    chunked_texts_output_file_path))
    
    chunked_texts = \
        ChunkedTexts(
            texts=texts,
            chunk_size=chunk_size,
            output_file_path=chunked_texts_output_file_path)

    if chunked_texts_output_file_path:
        texts.export_to_csv()
        
        chunked_texts.export_to_jsonl()

    return \
        chunked_texts
