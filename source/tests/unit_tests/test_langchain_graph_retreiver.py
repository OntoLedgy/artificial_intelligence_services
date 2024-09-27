# https://github.com/tomasonjo/blogs/blob/master/llm/ms_graphrag.ipynb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytest

from services.orchestrators.graph_rag_orchestrator import GraphRagOrchestrator
from services.tokenisation.tokeniser import num_tokens_from_string


class TestLangchainGraphRetriever:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.news = pd.read_csv(
            "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
        )

    def test_chunking_news_articles(self):
        self.news["tokens"] = [
            num_tokens_from_string(f"{row['title']} {row['text']}")
            for i, row in self.news.iterrows()
        ]

        self.news.head()

        sns.histplot(self.news["tokens"], kde=False)
        plt.title("Distribution of chunk sizes")
        plt.xlabel("Token count")
        plt.ylabel("Frequency")
        plt.show()

    def test_orchestrate_graph_rag(self):
        graph_rag_orchestrator = GraphRagOrchestrator(self.news)

        graph_rag_orchestrator.orchestrate()
