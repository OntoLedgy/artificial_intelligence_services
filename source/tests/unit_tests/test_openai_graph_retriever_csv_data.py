# https://github.com/tomasonjo/blogs/blob/master/llm/ms_graphrag.ipynb

import pytest

from services.graph_rag.orchestrators.knowledge_graph_from_tabular_dataset_orchestrator import orchestrate_retrieve_knowledge_graph_from_tabular_data_set


class TestLangchainGraphRetriever:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        pass

    def test_orchestrate_graph_rag(
            self,
            news_data):
        
        news_data = news_data[:5]
        
        knowledge_graph = orchestrate_retrieve_knowledge_graph_from_tabular_data_set(
                news_data)
        
        print (knowledge_graph)
