import os
from pathlib import Path
from networkx.readwrite.graphml import write_graphml
from source.b_code.configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
)
from source.b_code.services.orchestrators.graph_rag_orchestrator_boro_version import (
    BoroGraphRagOrchestrator,
)
from source.b_code.services.summarisation.pdf_summariser import PDFSummarizer
from source.z_sandpit.oxi.helpers.nf_open_ai_configurations_overrider_oxi import (
    override_nf_open_ai_configurations_oxi,
)
from source.z_sandpit.test_data.configuration.z_sandpit_test_constants import (
    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
    COMPACT_TIMESTAMP_SUFFIX,
)


if __name__ == "__main__":
    override_nf_open_ai_configurations_oxi()

    pdf_file_name = r"STIDS 2024 - Formalizing Informational Intelligence Uncertainty - v0.038 CPa - final format.pdf"

    pdf_path = os.path.join(Z_SANDPIT_TEST_DATA_FOLDER_PATH, "inputs", pdf_file_name)

    summarizer = PDFSummarizer(
        pdf_path=pdf_path, openai_api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY
    )

    pdf_full_text = summarizer.load_pdf_with_pdfplumber()

    # summarizer.load_and_split_pdf()

    graph_rag_orchestrator = BoroGraphRagOrchestrator(data_set=pdf_full_text)

    graph_documents = graph_rag_orchestrator.process_text(
        text=pdf_full_text, llm_transformer=graph_rag_orchestrator.llm_transformer
    )

    graph_rag_orchestrator.graph_documents = graph_documents

    networkx_articles_combined_graph = (
        graph_rag_orchestrator.get_combined_networkx_graph_from_graph_documents()
    )

    z_sandpit_folder_path = Path(__file__).parent.parent.__str__()

    output_file_path = os.path.join(
        Z_SANDPIT_TEST_DATA_FOLDER_PATH,
        "outputs",
        "STIDS2024-FormalizingInformationalIntelligenceUncertainty-v0038"
        + COMPACT_TIMESTAMP_SUFFIX
        + ".graphml",
    )

    write_graphml(networkx_articles_combined_graph, output_file_path)
