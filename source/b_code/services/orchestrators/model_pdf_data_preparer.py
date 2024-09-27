from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from services.data_preparation.pdf_services import extract_text_from_pdfs
from services.data_preparation.prepare_data import prepare_data_for_training
from services.orchestrators.list_of_dictionaries_to_json_file_writer import (
    write_list_of_dictionaries_to_json_file,
)


def prepare_model_pdf_data(
    pdf_folder_path: str,
    chunked_data_file_path: str = None,
    chunk_size: int = NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING,  # TODO: record this in the class
) -> list:
    pdf_texts = extract_text_from_pdfs(pdf_folder=pdf_folder_path)

    chunked_data = prepare_data_for_training(texts=pdf_texts, chunk_size=chunk_size)

    if chunked_data_file_path:
        write_list_of_dictionaries_to_json_file(
            output_file_path=chunked_data_file_path, list_of_dictionaries=chunked_data
        )

    return chunked_data
