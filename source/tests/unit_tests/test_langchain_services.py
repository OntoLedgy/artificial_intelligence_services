import openai
import os

from services.summarisation.pdf_summariser import PDFSummarizer

openai.api_key = os.getenv('OPENAI_API_KEY')


class TestLangChainDocumentSummariser:

    def test_connectivity(self):
        print(openai.api_key)
        models = openai.Model.list()
        for model in models['data']:
            print(model['id'])

    def test_summarize(self):
        pdf_path = r"C:\Users\khanm\Zotero\storage\ACKIDU9N\Kuper and Vardi - 1993 - The logical data model.pdf"
        openai_api_key = os.getenv('OPENAI_API_KEY')

        summarizer = PDFSummarizer(
            pdf_path,
            openai_api_key)

        # summarizer.load_and_split_pdf()
        # summary = summarizer.summarize()

        #print("Summary of the PDF:")
        #print(summary)
