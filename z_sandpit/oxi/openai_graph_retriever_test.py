import os
import pandas as pd
from networkx.readwrite.graphml import write_graphml

from services.graph_rag.orchestrators.knowledge_graph_from_tabular_dataset_orchestrator import get_combined_networkx_graph_from_graph_documents
from services.graph_rag.orchestrators.knowledge_graph_from_tabular_dataset_orchestrator import orchestrate_retrieve_knowledge_graph_from_tabular_data_set
# from ol_ai_services.b_code.services.orchestrators.knowledge_graph_rag_from_csv_orchestrator import (
#     BoroGraphRagOrchestrator,
# )
from source.z_sandpit.oxi.helpers.nf_open_ai_configurations_overrider_oxi import (
    override_nf_open_ai_configurations_oxi,
)
from source.z_sandpit.test_data.configuration.z_sandpit_test_constants import (
    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
)

if __name__ == "__main__":
    override_nf_open_ai_configurations_oxi()

    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )

    graph_documents = orchestrate_retrieve_knowledge_graph_from_tabular_data_set(news)

    # graph_rag_orchestrator.orchestrate()

    networkx_articles_combined_graph = (
        get_combined_networkx_graph_from_graph_documents(graph_documents)
    )

    output_file_path = os.path.join(
        Z_SANDPIT_TEST_DATA_FOLDER_PATH,
        "outputs",
        "openai_graph_retriever_test.graphml",
    )

    write_graphml(networkx_articles_combined_graph, output_file_path)
