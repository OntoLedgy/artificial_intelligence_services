from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from services.data_preparation.objects.chunked_texts import ChunkedTexts
from services.data_preparation.objects.texts import Texts


def get_chunked_texts(
        source_texts_folder_path: str,
        chunked_texts_output_file_path: str = None,
        chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING) \
    -> ChunkedTexts:
    texts = \
        Texts(
                source_texts_folder_path=source_texts_folder_path)
    
    chunked_texts = \
        ChunkedTexts(
                texts=texts,
                chunk_size=chunk_size)

    if chunked_texts_output_file_path:
        chunked_texts.export_to_jsonl(
                output_folder_path=chunked_texts_output_file_path)

    return \
        chunked_texts
