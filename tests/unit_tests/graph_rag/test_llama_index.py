# https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex

import os.path
import pytest
import logging
import sys
import pandas as pd
from typing import Literal

from llama_index.core import (
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    PropertyGraphIndex,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram

from graph_rag.domain_models.entities import Entities
from graph_rag.domain_models.persons import entities
from graph_rag.domain_models.persons import relations
from graph_rag.domain_models.persons import validation_schema

@pytest.mark.usefixtures("setup_tests")
class TestLlamaIndex:
    @pytest.fixture(autouse=True)
    def setup_tests(self, neo4j_config):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        
        # Setup Neo4j connection for graph tests using configuration from fixtures
        username = neo4j_config.get("USERNAME")
        password = neo4j_config.get("PASSWORD")
        url = neo4j_config.get("URI")
        
        print(f"Connecting to Neo4j at {url} with username {username}")
        
        try:
            self.graph_store = Neo4jPropertyGraphStore(
                username=username,
                password=password,
                url=url,
            )
            print("Neo4j connection successful")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            raise
        
        self.embed_model = OpenAIEmbedding()

    def test_llama_index(self):
        """
        Test basic VectorStoreIndex functionality with document loading,
        index creation/loading, and querying.
        """
        # Define paths
        persist_dir = "./data/outputs/vector_storage"
        input_dir = "./tests/data/inputs/txt"
        
        # Ensure input directory exists
        assert os.path.exists(input_dir), f"Input directory {input_dir} not found"
        
        # Create or load the index
        # Make sure the directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Check if index files exist
        docstore_path = os.path.join(persist_dir, "docstore.json")
        if not os.path.exists(docstore_path):
            # We already created the directory above
            
            # Load documents
            documents = SimpleDirectoryReader(input_dir).load_data()
            assert len(documents) > 0, "No documents loaded from input directory"
            
            # Create index
            index = VectorStoreIndex.from_documents(documents)
            assert index is not None, "Failed to create index from documents"
                    
            # Persist index
            index.storage_context.persist(persist_dir=persist_dir)
            assert os.path.exists(persist_dir), f"Failed to create persist directory {persist_dir}"
            
        else:
            # Load index from storage
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            assert storage_context is not None, "Failed to create storage context"
            
            index = load_index_from_storage(storage_context)
            assert index is not None, "Failed to load index from storage"

        # Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query("What did the author do growing up?")
        
        # Verify response
        assert response is not None, "Query response is None"
        assert len(str(response)) > 0, "Query response is empty"
        
        print(f"Response received: {response}")

    def test_llama_rag(self):
        """
        Test property graph index creation with Neo4j integration.
        This test includes entity and relation extraction from news articles.
        """
        # Load news data - use a Windows-compatible path
        news_csv_path = "./tests/data/inputs/news_articles.csv"
        
        # Download if not exists
        if not os.path.exists(news_csv_path):
            import urllib.request
            os.makedirs(os.path.dirname(news_csv_path), exist_ok=True)
            print(f"Downloading news articles CSV to {news_csv_path}")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv",
                news_csv_path
            )
            print("Download complete")
        
        # Verify file exists
        assert os.path.exists(news_csv_path), f"News CSV file not found at {news_csv_path}"
        
        # Load news articles
        try:
            news = pd.read_csv(news_csv_path)
            assert len(news) > 0, "No news articles found in CSV file"
        except Exception as e:
            pytest.fail(f"Failed to load news articles from CSV: {str(e)}")
        
        # Create document objects
        documents = [
            Document(text=f"{row['title']}: {row['text']}")
            for i, row in news.iterrows()
        ]
        assert len(documents) > 0, "Failed to create document objects from news articles"



        # Create knowledge graph extractor
        llm = OpenAI()
        
        kg_extractor = SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=entities,
            possible_relations=relations,
            kg_validation_schema=validation_schema,
            # if false, allows for values outside of the schema
            # useful for using the schema as a suggestion
            strict=True,
        )
        
        assert kg_extractor is not None, "Failed to create SchemaLLMPathExtractor"

        # Limit number of articles for test performance
        NUMBER_OF_ARTICLES = 20  # Using a smaller number for testing

        # Build property graph index
        try:
            index = PropertyGraphIndex.from_documents(
                documents[:NUMBER_OF_ARTICLES],
                kg_extractors=[kg_extractor],
                llm=llm,
                embed_model=self.embed_model,
                property_graph_store=self.graph_store,
                show_progress=True,
            )
            assert index is not None, "Failed to create PropertyGraphIndex"
        except Exception as e:
            pytest.fail(f"Error creating PropertyGraphIndex: {str(e)}")

        # Test basic querying functionality
        try:
            # Create a query engine
            query_engine = index.as_query_engine()
            
            # Execute a query
            response = query_engine.query(
                "What companies are mentioned and what are their relationships?"
            )
            assert response is not None, "Query response is None"
            assert len(str(response)) > 0, "Query response is empty"
            
            print(f"Graph RAG Query Response: {response}")
        except Exception as e:
            pytest.fail(f"Error querying PropertyGraphIndex: {str(e)}")
            
    def test_custom_graph_retriever(self):
        """
        Test creating and using a custom property graph retriever
        with specialized entity extraction and query capabilities.
        """
        
      
        # Test with minimal number of articles
        try:
            # Load a few news articles for this test
            news_csv_path = "./tests/data/inputs/news_articles.csv"
            if os.path.exists(news_csv_path):
                news = pd.read_csv(news_csv_path)
                # Convert to list before slicing
                documents = [
                    Document(text=f"{row['title']}: {row['text']}")
                    for i, row in list(news.iterrows())[:3]  # Just use 3 articles
                ]
                
                # Create basic graph extractor
                llm = OpenAI()
                entities = Literal["PERSON", "ORGANIZATION"]
                relations = Literal["WORKS_AT", "PARTNERSHIP"]
                
                kg_extractor = SchemaLLMPathExtractor(
                    llm=llm,
                    possible_entities=entities,
                    possible_relations=relations,
                    strict=False,
                )
                
                # Build a small index
                index = PropertyGraphIndex.from_documents(
                    documents,
                    kg_extractors=[kg_extractor],
                    llm=llm,
                    embed_model=self.embed_model,
                    property_graph_store=self.graph_store,
                    show_progress=True,
                )
                assert index is not None, "Failed to create PropertyGraphIndex"
                
                # Create and test the custom retriever
                custom_retriever = index.as_retriever(
                    include_text=True,
                )
                
                # Test the retriever with a query
                test_query = "What organisations are mentioned in the graph?"
                results = custom_retriever.retrieve(test_query)
                
                # Add assertions based on the expected structure
                assert results is not None, "Retriever returned None result"
                
                # Print results for debugging
                print(f"Custom retriever results:\n")
                for result in results:
                    print(f"\n{result}\n")
                
            else:
                pytest.skip("News CSV file not found, skipping custom retriever test")
                
        except Exception as e:
            pytest.fail(f"Error in custom graph retriever test: {str(e)}")
            
