import os
import pytest
from services.data_preparation.text_from_word_document_sections_extractor import (
    extract_text_from_word_document_sections,
)
from source.b_code.services.orchestrators.graph_rag_orchestrator_boro_version import (
    BoroGraphRagOrchestrator,
)
from source.z_sandpit.oxi.helpers.nf_open_ai_configurations_overrider_oxi import (
    override_nf_open_ai_configurations_oxi,
)
from source.z_sandpit.test_data.configuration.z_sandpit_test_constants import (
    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
    COMPACT_TIMESTAMP_SUFFIX,
)


# TODO: Test temporarily on hold
class TestOpenAiGraphRetrieverWordDocument:
    @pytest.fixture(autouse=True)
    def setup_method(self) -> None:
        override_nf_open_ai_configurations_oxi()

        word_document_file_name = r"STIDS 2024 - Formalizing Informational Intelligence Uncertainty - v0.038 CPa - final format - word document.docx"

        self.word_document_path = os.path.join(
            Z_SANDPIT_TEST_DATA_FOLDER_PATH, "inputs", word_document_file_name
        )

        if not os.path.isfile(self.word_document_path):
            raise FileNotFoundError

        self.output_file_path = os.path.join(
            Z_SANDPIT_TEST_DATA_FOLDER_PATH,
            "outputs",
            "STIDS2024-FormalizingInformationalIntelligenceUncertainty-v0038"
            + COMPACT_TIMESTAMP_SUFFIX
            + ".graphml",
        )

    def test_graph_retriever_pdf_document_full_text(self) -> None:
        word_document_sections = extract_text_from_word_document_sections(
            document_file_path=self.word_document_path
        )

        graph_rag_orchestrator = BoroGraphRagOrchestrator(
            data_set=word_document_sections
        )

        # graph_documents = \
        #     graph_rag_orchestrator.process_text(
        #         text=word_document_sections,
        #         llm_transformer=graph_rag_orchestrator.llm_transformer)
        #
        # graph_rag_orchestrator.graph_documents = \
        #     graph_documents
        #
        # networkx_graph = \
        #     graph_rag_orchestrator.get_combined_networkx_graph_from_graph_documents()
        #
        # write_graphml(
        #     networkx_graph,
        #     self.output_file_path)
