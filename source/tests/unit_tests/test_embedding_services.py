import os

import pytest
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

from code.common_utilities.pdf_services import load_pdfs

# Directory containing PDF articles
PDF_DIR = r'D:\OneDrives\OneDrive - OntoLedgy\Ontology of Money\Literature Review\Accounting'


def retrieve_similar_articles(query, model, index, articles, top_k=5, output_file="retrieve_similar_articles.txt"):
    # Create an embedding for the query
    query_embedding = model.encode([query], convert_to_tensor=False)

    # Search for similar articles in the index
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Retrieve the top-k articles
    retrieved_articles = [articles[i] for i in indices[0]]

    with open(output_file, 'w', encoding='utf-8') as file:
        for article in retrieved_articles:
            file.write(article + "\n---\n")  # Use "---" as a delimiter between articles

    return retrieved_articles

# Helper function to truncate context to fit within a token limit
def truncate_context(context, max_tokens=4000):
    # Truncate the context to the specified number of characters (approximation for tokens)
    return context[:max_tokens]
# Function to get response using retrieved articles


def get_response(
        query,
        client,
        model_name="gpt-3.5-turbo",
        input_file='retrieved_articles.txt',
        max_context_tokens=12000):

    # Read the retrieved articles from the file
    with open(input_file, 'r', encoding='utf-8') as file:
        context = file.read()

    # Truncate the context to fit within the max context tokens
    truncated_context = truncate_context(context, max_context_tokens)

    # Craft the prompt with query and context
    prompt = f"Context: {truncated_context}\n\nBased on the provided context, please answer the following question: {query}\nAnswer:"

    # Use the OpenAI client to generate a response
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return the generated answer
    return response.choices[0].message.content



class TestEbeddings:

    @pytest.fixture(autouse=True)
    def setup_method(self):

        # Load articles from PDFs
        self.articles = load_pdfs(PDF_DIR)
        # Load a pre-trained Sentence Transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.index_file_full_path = './data/outputs/embeddings/pdf_article_embeddings.index'
        self.file_metadata = './data/outputs/embeddings/pdf_article_texts.npy'
        self.output_file = './data/outputs/embeddings/retrieved_articles.txt'

        self.query = "explain the different types of accounting practices in the 17th century"
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


    def test_embeddings(self):


        # Generate embeddings for each article
        article_embeddings = self.model.encode(self.articles, convert_to_tensor=False)

        # Convert embeddings to a numpy array
        article_embeddings = np.array(article_embeddings)

        # Initialize FAISS index
        embedding_dimension = article_embeddings.shape[1]  # Dimension of embeddings

        index = faiss.IndexFlatL2(embedding_dimension)

        # Add embeddings to the index
        index.add(article_embeddings)

        # Save the index and article metadata for later use
        faiss.write_index(index,self.index_file_full_path )

        np.save(self.file_metadata , self.articles)


    def test_querying_embeddings(self):
        # Load the saved FAISS index from disk
        index = faiss.read_index(self.index_file_full_path)

        # Load the saved articles
        articles = np.load(self.file_metadata , allow_pickle=True)

        # Example query
        retrieved_articles = retrieve_similar_articles(
            self.query,
            self.model,
            index,
            articles,
            output_file=self.output_file)

        # Display retrieved articles
        for i, article in enumerate(retrieved_articles):
            print(f"Article {i + 1}:\n{article[:500]}...\n")  # Display first 500 characters for brevity


    def test_rag_respose(self):
        # Generate a response using the retrieved articles as context
        response = get_response(
            self.query,
            client=self.client,
            input_file=self.output_file)

        print(response)




