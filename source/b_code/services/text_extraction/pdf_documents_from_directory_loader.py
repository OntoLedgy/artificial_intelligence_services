import glob
import os
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import (
    run_and_log_function,
)
from configurations.constants import PDF_FILE_EXTENSION
from services.text_extraction.text_from_pdf_document_extractor import extract_text_from_pdf


@run_and_log_function()
def load_pdf_documents_from_directory(
    directory_path: str, looks_into_subfolders: bool = True
) -> list:
    pdf_document_texts = list()

    recursive_globs_pattern = "/**/*" if looks_into_subfolders else "/*"

    file_system_object_paths = glob.glob(
        pathname=directory_path + recursive_globs_pattern + PDF_FILE_EXTENSION,
        recursive=True,
    )

    for file_system_object_path in file_system_object_paths:
        pdf_document_texts = __add_pdf_text(
            file_system_object_path=file_system_object_path,
            pdf_document_texts=pdf_document_texts,
        )

    return pdf_document_texts


@run_and_log_function()
def __add_pdf_text(file_system_object_path: str, pdf_document_texts: list) -> list:
    if not file_system_object_path.endswith(PDF_FILE_EXTENSION):
        return pdf_document_texts

    if not os.path.isdir(file_system_object_path):
        pdf_text = extract_text_from_pdf(pdf_path=file_system_object_path)

        pdf_document_texts.append(pdf_text)

    return pdf_document_texts
