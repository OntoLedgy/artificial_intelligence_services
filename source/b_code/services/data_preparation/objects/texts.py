import os
import PyPDF2
from configurations.constants import PDF_FILE_EXTENSION
from configurations.constants import READ_BYTES_ACRONYM


class Texts:
    def __init__(
            self,
            source_texts_folder_path: str):
        self.source_texts_folder_path = \
            source_texts_folder_path
        
        self.source_texts = \
            self.load_pdf_source_texts()
    
    def load_pdf_source_texts(
            self) \
            -> list:
        source_texts = \
            list()
        
        for filename \
                in os.listdir(
                    self.source_texts_folder_path):
            if filename.endswith(PDF_FILE_EXTENSION):
                self.__extract_texts_from_pdf(
                    filename=filename,
                    source_texts=source_texts)
        
        return \
            source_texts
    
    
    def __extract_texts_from_pdf(
            self,
            filename: str,
            source_texts: list):
        file_path = \
            os.path.join(
                    self.source_texts_folder_path,
                    filename)
        text = \
            self.extract_text_from_pdf(
                    pdf_path=file_path)
        
        source_texts.append(
                text)
    
    @staticmethod
    def extract_text_from_pdf(
            pdf_path: str):
        text = \
            str()
        
        with open(
                pdf_path,
                READ_BYTES_ACRONYM) as file:
            pdf_reader = \
                PyPDF2.PdfReader(
                    file)
            
            for page_number \
                    in range(len(pdf_reader.pages)):
                pdf_page = \
                    pdf_reader.pages[page_number]
                
                text += \
                    pdf_page.extract_text()
                
        return \
            text
    
    # TODO: Create an export to csv method for inspection??
    
    # TODO: check the type is a string
    