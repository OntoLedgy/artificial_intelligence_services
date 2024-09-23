import os
import PyPDF2


def extract_text_from_pdfs(pdf_folder):
    pdf_texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            with open(os.path.join(pdf_folder, pdf_file), 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
                pdf_texts.append(text)
    return pdf_texts


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


# Read articles from PDFs into a list
def load_pdfs(directory):
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(file_path)
            pdf_texts.append(text)
    return pdf_texts
