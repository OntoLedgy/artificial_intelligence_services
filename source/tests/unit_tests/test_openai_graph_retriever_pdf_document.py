import os
import pytest
from networkx.readwrite.graphml import write_graphml
from nf_common_source.code.services.reporting_service.reporters.log_file import LogFiles

from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
    )
from configurations.constants import GRAPHML_FILE_EXTENSION
from services.text_extraction.pdf_documents_from_directory_loader import (
    load_pdf_documents_from_directory,
    )
from services.orchestrators.retrieve_knowledge_graph_from_text_file_orchestrator import orchestrate_retrieve_knowledge_graph_from_text_file
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


class TestOpenAiGraphRetrieverPdfDocument:
    
    @pytest.fixture(
            autouse=True)
    def setup_method(
            self,
            test_data_folder_absolute_path,
            inputs_folder_absolute_path,
            outputs_folder_absolute_path) -> None:
        override_nf_open_ai_configurations_oxi()
        
        pdf_file_name = (r"STIDS 2024 - Formalizing Informational Intelligence Uncertainty - v0.038 CPa - final "
                         r"format.pdf")
        
        self.pdf_path = os.path.join(
                inputs_folder_absolute_path,
                "pdf/bclearer",
                pdf_file_name
                )
        
        self.output_file_path = os.path.join(
                outputs_folder_absolute_path,
                "graph_rag/bclearer",
                "STIDS2024-FormalizingInformationalIntelligenceUncertainty-v0038"
                + COMPACT_TIMESTAMP_SUFFIX
                + GRAPHML_FILE_EXTENSION,
                )
        
        LogFiles.open_log_file(
                folder_path=os.path.join(
                    test_data_folder_absolute_path,
                    "logs")
                )
    
    # TODO: Can this test be parametrised with @pytest.mark.parametrize to test different model and temperatures
    #  at once?
    def test_graph_retriever_pdf_document_full_text(
            self) \
            -> None:
        knowledge_directed_graph = \
            orchestrate_retrieve_knowledge_graph_from_text_file(
                text_file_path=self.pdf_path,
                model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI,
                temperature=NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE)
        
        write_graphml(
            knowledge_directed_graph,
            self.output_file_path)
    
    
    # TODO: test directory of pdfs - legacy not adapted
    def test_graph_retriever_pdf_directory_with_subfolders(
            self,
            inputs_folder_absolute_path) -> None:
        zotero_pdf_directory_test_folder_form_path = os.path.join(
                inputs_folder_absolute_path,
                "pdf/bclearer/STIDS-2024-bCLEARer_form"
                )
        
        pdf_documents = load_pdf_documents_from_directory(
                directory_path=zotero_pdf_directory_test_folder_form_path,
                looks_into_subfolders=True,
                )
        
        graph_rag_orchestrator = BoroGraphRagOrchestrator(
                model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI,
                data_set=pdf_documents,
                )
        
        # TODO: Try to orchestrate sequentially rather than in parallel to test if the error caused by too many tokens
        #  being created still persists
        graph_rag_orchestrator.orchestrate()
        
        networkx_graph = (
            graph_rag_orchestrator.get_combined_networkx_graph_from_graph_documents()
        )
        
        write_graphml(
            networkx_graph,
            self.output_file_path)
