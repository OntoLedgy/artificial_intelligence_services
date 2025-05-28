#https://www.marktechpost.com/2024/12/03/meet-megaparse-an-open-source-ai-tool-for-parsing-various-types-of-documents-for-llm-ingestion/
from megaparse import MegaParse
from langchain_openai import ChatOpenAI

import os

def parse_document(
        document_path,
        parsed_document_path):
    # Initialize the language model
    model = ChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"))
    
    # Set up the parser

    document_parser = MegaParse()
    document_parser.unstructured_parser.model = model
    # Load and process the document
    response = document_parser.load(document_path)
    print("document parsed, final output -----\n")
    print(response)
    
    

