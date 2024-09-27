import os
import PyPDF2

from configurations.constants import PDF_FILE_EXTENSION
from configurations.constants import READ_BYTES_ACRONYM


# TODO: MKh - these first two methods appear the same? Merge?
# TODO: MKh - should these be separate files? Helpers?
# TODO: MKh - should these be clean coded?
def extract_text_from_pdfs(
        pdf_folder):
    pdf_texts = []
    for pdf_file in os.listdir(
            pdf_folder):
        if pdf_file.endswith(
                PDF_FILE_EXTENSION):
            with open(
                    os.path.join(
                            pdf_folder,
                            pdf_file),
                    READ_BYTES_ACRONYM) as file:
                reader = PyPDF2.PdfReader(
                    file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                pdf_texts.append(
                    text)
    return pdf_texts


# TODO: MKh - should 'rb' be a common literal?
def extract_text_from_pdf(
        pdf_path):
    text = ""
    with open(
            pdf_path,
            READ_BYTES_ACRONYM) as file:
        pdf_reader = PyPDF2.PdfReader(
            file)
        for page_num in range(
                len(
                        pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


# TODO: MKh - should ".pdf" be a common literal?
# Read articles from PDFs into a list
def load_pdfs(
        directory):
    pdf_texts = []
    for filename in os.listdir(
            directory):
        if filename.endswith(
                PDF_FILE_EXTENSION):
            file_path = os.path.join(
                directory,
                filename)
            text = extract_text_from_pdf(
                file_path)
            pdf_texts.append(
                text)
    return pdf_texts
