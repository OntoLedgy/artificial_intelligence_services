import os
import pytest
from networkx.readwrite.graphml import write_graphml

from services.graph_rag.orchestrators.knowledge_graph_from_word_file_orchestrator import orchestrate_retrieve_knowledge_graph_from_word_file
from services.graph_rag.transformers.networkx_from_graph_document_getter import get_combined_networkx_graph_from_graph_documents

from tests.data.configuration.z_sandpit_test_constants import (
    COMPACT_TIMESTAMP_SUFFIX,
)


class TestOpenAiGraphRetrieverWordDocument:
    @pytest.fixture(autouse=True)
    def setup_method(
            self,
            inputs_folder_absolute_path,
            outputs_folder_absolute_path) -> None:
        
        word_document_file_name = r"STIDS 2024 - Formalizing Informational Intelligence Uncertainty - v0.038 CPa - final format - word document.docx"

        self.word_document_path = os.path.join(
            inputs_folder_absolute_path,
            "word",
            word_document_file_name
        )

        if not os.path.isfile(self.word_document_path):
            raise FileNotFoundError

        self.output_file_path = os.path.join(
            outputs_folder_absolute_path,
            "graph_rag/bclearer",
            "STIDS2024-FormalizingInformationalIntelligenceUncertainty-v0038"
            + COMPACT_TIMESTAMP_SUFFIX
            + ".graphml",
        )

    def test_graph_retriever_word_document_full_text(self) -> None:
        
        graph_documents = orchestrate_retrieve_knowledge_graph_from_word_file(
                word_file_path=self.word_document_path
                )
        
        knowledge_directed_graph = get_combined_networkx_graph_from_graph_documents(
            graph_documents)
        
        write_graphml(
            knowledge_directed_graph,
            self.output_file_path )
