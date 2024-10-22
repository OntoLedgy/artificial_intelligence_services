import glob
import os
from pandas import DataFrame
from typing import Union

import PyPDF2
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import (
    run_and_log_function,
)
from configurations.constants import PDF_FILE_EXTENSION
from configurations.constants import READ_BYTES_ACRONYM
from services.text_extraction.pdf_document_extractor import extract_text_from_pdf


@run_and_log_function()
def extract_dataframe_from_pdfs_in_folder(
        directory_path: Union[str, bytes],
        looks_into_subfolders: bool = True
        ) -> DataFrame:
    
    titles = []
    texts = []
    
    # Construct the search pattern without leading slashes
    if looks_into_subfolders:
        search_pattern = os.path.join(directory_path, '**', f'*{PDF_FILE_EXTENSION}')
    else:
        search_pattern = os.path.join(directory_path, f'*{PDF_FILE_EXTENSION}')

    
    file_system_object_paths = glob.glob(
            pathname=search_pattern,
            recursive=True,
            )
    
    for file_system_object_path in file_system_object_paths:
        
        title = os.path.splitext(
            os.path.basename(
                file_system_object_path))[0]

        text = extract_text_from_pdf(
            file_system_object_path)
        
        titles.append(
            title)
        texts.append(
            text)
    
    data = {
        'title': titles,
        'text' : texts
        }
    
    df = DataFrame(
        data)
    
    return df


# TODO: MKh - should these be separate files? Helpers?
# TODO: MKh - should these be clean coded?
# TODO: MKh - See class Texts
def extract_text_from_pdfs_in_folder(
        directory)->list[str]:
    
    pdf_texts = []
    
    for pdf_file in os.listdir(
            directory):
        
        if pdf_file.endswith(
                PDF_FILE_EXTENSION):
            with open(
                    os.path.join(
                            directory,
                            pdf_file),
                    READ_BYTES_ACRONYM) as file:
                
                text = extract_text_from_pdf(
                        file)
                
                pdf_texts.append(
                        text)
    return pdf_texts