import os
import pytest
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite.graphml import write_graphml
from nf_common_source.code.services.reporting_service.reporters.log_file import LogFiles

from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
    )
from configurations.constants import GRAPHML_FILE_EXTENSION
from services.graph_rag.orchestrators.knowledge_graph_rag_from_csv_orchestrator import get_combined_networkx_graph_from_graph_documents
from services.graph_rag.orchestrators.knowledge_graph_rag_from_csv_orchestrator import orchestrate_graph_rag_from_csv
from services.text_extraction.pdf_documents_from_directory_loader import (
    load_pdf_documents_from_directory,
    )
from services.graph_rag.orchestrators.knowledge_graph_rag_from_text_file_orchestrator import orchestrate_retrieve_knowledge_graph_from_text_file
from source.z_sandpit.test_data.configuration.z_sandpit_test_constants import (
    COMPACT_TIMESTAMP_SUFFIX,
    )


class TestOpenAiGraphRetrieverPdfDocument:
    
    @pytest.fixture(
            autouse=True)
    def setup_method(
            self,
            pdf_file_path,
            pdf_folder_path,
            test_data_folder_absolute_path,
            inputs_folder_absolute_path,
            outputs_folder_absolute_path) -> None:
            
        #TODO - user specific overrides should not be done in tests, (can use user_specific configurations in relevant configuration json).
        # override_nf_open_ai_configurations_oxi()
       
        self.pdf_path = pdf_file_path
        
        self.pdf_folder_path = pdf_folder_path
        
        self.output_file_base_name = os.path.basename(
                pdf_file_path)
        
        self.outputs_folder_path = os.path.join(
                outputs_folder_absolute_path,
                "graph_rag/bclearer"
                )
        
        self.folder_graph_file_path = os.path.join(
                self.outputs_folder_path,
                os.path.basename(pdf_folder_path)
                + COMPACT_TIMESTAMP_SUFFIX
                + GRAPHML_FILE_EXTENSION
                )
        
        self.single_pdf_graph_file_path = os.path.join(
                self.outputs_folder_path,
                self.output_file_base_name
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
            self.single_pdf_graph_file_path)
    
    
    # TODO: test directory of pdfs - legacy not adapted  - ADAPTED - DONE
    def test_graph_retriever_pdf_directory_with_subfolders(
            self
            ) -> None:
        
        
        pdf_documents = load_pdf_documents_from_directory(
                directory_path = self.pdf_folder_path,
                looks_into_subfolders=True,
                )
        
        # TODO: Try to orchestrate sequentially rather than in parallel to test if the error caused by too many tokens
        #  being created still persists - fixed with RateLimits implementation - DONE
        
        graph_documents = orchestrate_graph_rag_from_csv(
                model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O,
                data_set=pdf_documents,
                )
        
        networkx_graph = (
            get_combined_networkx_graph_from_graph_documents(
                    graph_documents)
        )
        
        write_graphml(
            networkx_graph,
            self.folder_graph_file_path)
        
    def test_visualise_graph(self):

        graph = nx.read_graphml(
            self.folder_graph_file_path)
        
        # Draw the graph
        plt.figure(
            figsize=(10, 10))
        nx.draw(
            graph,
            with_labels=True,
            node_color='skyblue',
            font_size=10,
            node_size=1000,
            edge_color='gray')
        plt.show()
