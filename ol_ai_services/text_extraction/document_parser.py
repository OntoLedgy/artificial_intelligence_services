#https://www.marktechpost.com/2024/12/03/meet-megaparse-an-open-source-ai-tool-for-parsing-various-types-of-documents-for-llm-ingestion/
from megaparse.core.megaparse import MegaParse
from langchain_openai import ChatOpenAI
from megaparse.core.parser.unstructured_parser import UnstructuredParser
import os


def parse_document(
        document_path,
        parsed_document_path):
    # Initialize the language model
    model = ChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"))
    
    # Set up the parser
    parser = UnstructuredParser(model=model)
    megaparse = MegaParse(parser)
    
    # Load and process the document
    response = megaparse.load(document_path)
    print(response)
    
    # Save the processed content to a markdown file
    megaparse.save(parsed_document_path)