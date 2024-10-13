# https://github.com/tomasonjo/blogs/blob/master/llm/ms_graphrag.ipynb

import pytest


from services.orchestrators.knowledge_graph_rag_from_csv_orchestrator import orchestrate_graph_rag_from_csv


class TestLangchainGraphRetriever:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        pass

    def test_orchestrate_graph_rag(
            self,
            news_data):
        
        knowledge_graph = orchestrate_graph_rag_from_csv(
                news_data)
        
        print (knowledge_graph)
