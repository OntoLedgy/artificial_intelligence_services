import openai
import os
import pytest

from services.summarisation.pdf_summariser import PDFSummarizer

openai.api_key = os.getenv("OPENAI_API_KEY")


class TestLangChainDocumentSummariser:
    @pytest.fixture(autouse=True)
    def setup_method(self,
                     inputs_folder_absolute_path):
        
        self.pdf_path = os.path.join(
                inputs_folder_absolute_path,
                "pdf/accounting/A reference ontology of money and virtual currencies.pdf"
                )

    def test_connectivity(self):
        print(openai.api_key)
        models = openai.models.list()
        for model in models:
            print(model.id)

    def test_summarize(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")

        summarizer = PDFSummarizer(
                self.pdf_path,
                openai_api_key)

        summarizer.load_and_split_pdf()
        summary = summarizer.summarize()

        print("Summary of the PDF:")
        print(summary)
