import os
import pytest
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from services.data_preparation.pdf_services import load_pdfs
from services.embeddings.embeddings import Embeddings
from services.embeddings.search_embedded_documents import (
    retrieve_similar_documents,
    get_response,
)

PDF_DIR = r"./data/inputs/pdf"


class TestEmbeddings:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Load articles from PDFs
        self.articles = load_pdfs(PDF_DIR)
        # Load a pre-trained Sentence Transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index_file_full_path = (
            "./data/outputs/embeddings/pdf_article_embeddings.index"
        )
        self.file_metadata = "./data/outputs/embeddings/pdf_article_texts.npy"
        self.output_file = "./data/outputs/embeddings/retrieved_articles.txt"
        self.query = "describe different types of ontologies in computing"

    def test_embeddings(self):
        embedding = Embeddings(
            model=self.model,
            documents=self.articles,
            index_file_full_path=self.index_file_full_path,
            file_metadata=self.file_metadata,
        )

        embedding.create()
        embedding.save()

    def test_querying_embeddings(self):
        # Load the saved FAISS index from disk
        index = faiss.read_index(self.index_file_full_path)

        articles = np.load(self.file_metadata, allow_pickle=True)

        retrieved_articles = retrieve_similar_documents(
            self.query, self.model, index, articles, output_file=self.output_file
        )

        for i, article in enumerate(retrieved_articles):
            print(
                f"Article {i + 1}:\n{article[:500]}...\n"
            )  # Display first 500 characters for brevity

    def test_rag_response(self):
        # Generate a response using the retrieved articles as context
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = get_response(
            self.query, client=self.client, input_file=self.output_file
        )

        print(response)
