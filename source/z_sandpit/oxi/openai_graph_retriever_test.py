import pandas as pd
from source.code.services.orchestrators.graph_rag_orchestrator import GraphRagOrchestrator


if __name__ == '__main__':
    news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")

    graph_rag_orchestrator = GraphRagOrchestrator(news)

    graph_rag_orchestrator.orchestrate()
