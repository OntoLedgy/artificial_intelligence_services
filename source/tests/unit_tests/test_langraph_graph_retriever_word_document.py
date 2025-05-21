import os
import pytest
from langchain_core.messages import HumanMessage

from services.graph_rag.orchestrators.knowledge_graph_langgraph_extractor import run_knowledge_graph_pipeline
from services.text_extraction.word_document_sections_extractor import extract_text_from_word_document_sections

from source.z_sandpit.test_data.configuration.z_sandpit_test_constants import (
    COMPACT_TIMESTAMP_SUFFIX,
)
from bclearer_interop_services.graph_services.visualisation_service.graph_visualiser import visualize_graph

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
        
        topic = "information uncertainty"
        
        word_document_sections = extract_text_from_word_document_sections(
                document_file_path=self.word_document_path
                )
        
        knowledge_graph_state = {
            "topic"             : topic,
            "raw_text"          : word_document_sections[1],
            "entities"          : [],
            "relations"         : [],
            "resolved_relations": [],
            "graph"             : None,
            "validation"        : {},
            "messages"          : [HumanMessage(
                content=f"Build a knowledge graph about {topic}")],
            "current_agent"     : "data_gatherer"
            }
        
        result = run_knowledge_graph_pipeline(
                topic,
                knowledge_graph_state)
        
        visualize_graph(
                result["graph"])
