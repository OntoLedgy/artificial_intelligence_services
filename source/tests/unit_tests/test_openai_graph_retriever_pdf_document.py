import os
import pytest
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite.graphml import write_graphml
from nf_common_source.code.services.reporting_service.reporters.log_file import LogFiles

from configurations.constants import GRAPHML_FILE_EXTENSION
from services.graph_rag.orchestrators.knowledge_graph_from_pdf_folder_orchestrator import orchestrate_retrieve_knowledge_graph_from_pdf_folder_file

from services.graph_rag.orchestrators.knowledge_graph_from_text_file_orchestrator import orchestrate_retrieve_knowledge_graph_from_text_file
from services.graph_rag.transformers.networkx_from_graph_document_getter import get_combined_networkx_graph_from_graph_documents
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
                text_file_path=self.pdf_path
                    )
        
        write_graphml(
            knowledge_directed_graph,
            self.single_pdf_graph_file_path)
    
    
    def test_graph_retriever_pdf_directory_with_subfolders(
            self
            ) -> None:
      
        graph_documents = orchestrate_retrieve_knowledge_graph_from_pdf_folder_file(
                folder_path=self.pdf_folder_path
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
