import PyPDF2
from configurations.constants import READ_BYTES_ACRONYM

def extract_text_from_pdf(
        pdf_path
        )->str:
    
    text = ""
    with open(
            pdf_path,
            READ_BYTES_ACRONYM) as file:
        
        pdf_reader = PyPDF2.PdfReader(
                file)
        
        for page_num in range(
                len(
                        pdf_reader.pages)):
            
            page = \
                pdf_reader.pages[page_num]
            
            text += \
                page.extract_text()
            
        extracted_text = text
        
    return extracted_text


