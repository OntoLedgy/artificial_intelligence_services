import openai
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from pypdf.errors import PdfReadError

from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations

# OpenAI API key setup
# openai.api_key = os.getenv('OPENAI_API_KEY')


class PDFSummarizer:
    def __init__(
            self,
            pdf_path: str,
            openai_api_key):

        self.pdf_path = pdf_path

        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI,
            temperature=NfOpenAiConfigurations.OPEN_AI_TEMPERATURE,
            max_tokens=NfOpenAiConfigurations.OPEN_AI_MAX_TOKENS
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=NfGeneralConfigurations.RECURSIVE_CHARACTER_TEXTSPLITTER_CHUNK_SIZE,
            chunk_overlap=NfGeneralConfigurations.RECURSIVE_CHARACTER_TEXTSPLITTER_CHUNK_OVERLAP)

        self.documents = None

        self.summarizer_chain = None

    def load_pdf_with_pypdf(self):

        try:
            from langchain.document_loaders import PyPDFLoader

            loader = PyPDFLoader(
                self.pdf_path)

            return loader.load_and_split(
                self.text_splitter)

        except PdfReadError as e:
            print(f"Error using PyPDFLoader: {e}")
            return None

    def load_pdf_with_pdfplumber(self):

        try:
            with pdfplumber.open(
                    self.pdf_path) as pdf:

                full_text = ""

                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"

                return full_text

        except Exception as e:
            print(
                f"Error using pdfplumber: {e}")

            return None

    def load_and_split_pdf(self):

        # Try loading with PyPDFLoader first
        self.documents = self.load_pdf_with_pypdf()

        # If PyPDFLoader fails, try pdfplumber
        if not self.documents:
            print("Falling back to pdfplumber...")
            text = self.load_pdf_with_pdfplumber()
            if text:
                self.documents = self.text_splitter.create_documents(
                    [text])

    def create_summary_chain(self):

        self.summarizer_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce")

    def summarize(self):

        if not self.documents:
            raise ValueError(
                "No documents loaded. Call load_and_split_pdf() first.")

        if not self.summarizer_chain:
            self.create_summary_chain()

        summary = self.summarizer_chain.run(
            self.documents)

        return summary
