from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from configurations.constants import PDF_FILE_EXTENSION
from services.data_preparation.objects.chunked_texts import ChunkedTexts
from services.data_preparation.objects.texts import Texts
from services.data_preparation.pdf_services import load_pdfs


def get_chunked_texts(
        source_texts_folder_path: str,
        chunked_texts_output_file_path: str = None,
        extension: str = PDF_FILE_EXTENSION,
        chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING) \
    -> ChunkedTexts:
    # TODO: Only PDF implemented at the moment
    if extension == PDF_FILE_EXTENSION:
        source_texts = \
            load_pdfs(
                    directory=source_texts_folder_path)
    
    else:
        source_texts = \
            list()
        
    texts = \
        Texts(
            source_texts=source_texts)
    
    chunked_texts = \
        ChunkedTexts(
            texts=texts,
            chunk_size=chunk_size,
            output_file_path=chunked_texts_output_file_path)

    if chunked_texts_output_file_path:
        chunked_texts.export_to_jsonl()  # TODO: move the output folder to the class parameters - DONE

    return \
        chunked_texts
