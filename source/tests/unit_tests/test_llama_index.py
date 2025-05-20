# https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex

from llama_index.core import (
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding

import os.path
import pytest
import logging
import sys
import pandas as pd
from typing import Literal


@pytest.mark.usefixtures("setup_tests")
class TestLlamaIndex:
    @pytest.fixture(autouse=True)
    def setup_tests(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        username = "neo4j"
        password = "stump-inlet-student"
        url = "bolt://localhost:7687"

        self.graph_store = Neo4jPGStore(
            username=username,
            password=password,
            url=url,
        )

        self.embed_model = OpenAIEmbedding()

    def test_llama_index(self):
        persist_dir = "./data/outputs/vector_storage"
        if not os.path.exists(persist_dir):
            
            documents = SimpleDirectoryReader(
                    "./data/inputs/txt").load_data()
            
            index = VectorStoreIndex.from_documents(
                    documents)
                    
            index.storage_context.persist(
                    persist_dir=persist_dir)
            
        else:
            
            storage_context = StorageContext.from_defaults(
                    persist_dir=persist_dir)
            
            index = load_index_from_storage(
                    storage_context)

        # Either way we can now query the index
        query_engine = index.as_query_engine()
        response = query_engine.query("What did the author do growing up?")
        print(response)

    def test_llama_rag(self):
        news = pd.read_csv(
            "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
        )
        documents = [
            Document(text=f"{row['title']}: {row['text']}")
            for i, row in news.iterrows()
        ]

        entities = Literal[
            "PERSON",
            "LOCATION",
            "ORGANIZATION",
            "PRODUCT",
            "EVENT"]
        
        relations = Literal[
            "SUPPLIER_OF",
            "COMPETITOR",
            "PARTNERSHIP",
            "ACQUISITION",
            "WORKS_AT",
            "SUBSIDIARY",
            "BOARD_MEMBER",
            "CEO",
            "PROVIDES",
            "HAS_EVENT",
            "IN_LOCATION",
        ]

        # define which entities can have which relations
        validation_schema = {
            "Person": [
                "WORKS_AT",
                "BOARD_MEMBER",
                "CEO",
                "HAS_EVENT"],
            "Organization": [
                "SUPPLIER_OF",
                "COMPETITOR",
                "PARTNERSHIP",
                "ACQUISITION",
                "WORKS_AT",
                "SUBSIDIARY",
                "BOARD_MEMBER",
                "CEO",
                "PROVIDES",
                "HAS_EVENT",
                "IN_LOCATION",
            ],
            "Product": ["PROVIDES"],
            "Event": ["HAS_EVENT", "IN_LOCATION"],
            "Location": ["HAPPENED_AT", "IN_LOCATION"],
        }

        kg_extractor = SchemaLLMPathExtractor(
            llm=OpenAI(),
            possible_entities=entities,
            possible_relations=relations,
            kg_validation_schema=validation_schema,
            # if false, allows for values outside of the schema
            # useful for using the schema as a suggestion
            strict=True,
        )

        NUMBER_OF_ARTICLES = 250

        index = PropertyGraphIndex.from_documents(
            documents[:NUMBER_OF_ARTICLES],
            kg_extractors=[kg_extractor],
            llm=OpenAI(),
            embed_model=self.embed_model,
            property_graph_store=self.graph_store,
            show_progress=True,
        )

        print(index)
