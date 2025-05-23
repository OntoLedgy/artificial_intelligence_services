import os
import pytest
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from services.embeddings.objects.embeddings import Embeddings
from services.embeddings.search_embedded_documents import (
    retrieve_similar_documents,
    get_response_using_retrieved_documents,
)
from services.text_extraction.pdf_folder_extractor import extract_text_from_pdfs_in_folder


class TestEmbeddings:
    @pytest.fixture(autouse=True)
    def setup_method(self,
                     outputs_folder_absolute_path,
                     inputs_folder_absolute_path):
       
        
        input_pdf_directory = os.path.join(
                inputs_folder_absolute_path,
                "pdf/accounting")
        
        self.articles = extract_text_from_pdfs_in_folder(
                input_pdf_directory)
        
        self.model = SentenceTransformer(
                "all-MiniLM-L6-v2")

        self.index_file_full_path = os.path.join(
                outputs_folder_absolute_path,
                "embeddings/pdf_article_embeddings.index"
                )
        
        self.file_metadata = os.path.join(
                outputs_folder_absolute_path,
                "embeddings/pdf_article_texts.npy"
                )
        self.retrieved_articles_text_file_path = os.path.join(
                outputs_folder_absolute_path,
                "embeddings/retrieved_articles.txt"
                )
        
        self.query = \
            "describe different types of ontologies in computing"

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
        
        index = faiss.read_index(
                self.index_file_full_path)

        articles = np.load(
                self.file_metadata,
                allow_pickle=True)

        retrieved_articles = retrieve_similar_documents(
            self.query,
            self.model,
                index,
                articles,
                output_file=self.retrieved_articles_text_file_path
        )

        for index, article in enumerate(
                retrieved_articles):
            print(
                f"Article {index + 1}:\n{article[:500]}...\n"
            )

    def test_rag_response(self):

        response = get_response_using_retrieved_documents(
            self.query,
            input_file=self.retrieved_articles_text_file_path
        )

        print(response)
