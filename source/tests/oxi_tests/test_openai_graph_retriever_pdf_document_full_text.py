import os
from pathlib import Path

import pytest
from networkx.readwrite.graphml import write_graphml
from source.b_code.configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from source.b_code.services.orchestrators.graph_rag_orchestrator_boro_version import BoroGraphRagOrchestrator
from source.b_code.services.summarisation.pdf_summariser import PDFSummarizer
from source.z_sandpit.oxi.helpers.nf_open_ai_configurations_overrider_oxi import override_nf_open_ai_configurations_oxi
from source.z_sandpit.test_data.configuration.z_sandpit_test_constants import Z_SANDPIT_TEST_DATA_FOLDER_PATH, \
    COMPACT_TIMESTAMP_SUFFIX


class TestOpenAiGraphRetrieverPdfDocumentFullText:
    @pytest.fixture(
            autouse=True)
    def setup_method(
            self) \
            -> None:
        override_nf_open_ai_configurations_oxi()
        
        pdf_file_name = \
            r'STIDS 2024 - Formalizing Informational Intelligence Uncertainty - v0.038 CPa - final format.pdf'
        
        self.pdf_path = \
            os.path.join(
                    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
                    'inputs',
                    pdf_file_name)
        
        self.output_file_path = \
            os.path.join(
                    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
                    'outputs',
                    'STIDS2024-FormalizingInformationalIntelligenceUncertainty-v0038' + COMPACT_TIMESTAMP_SUFFIX + '.graphml')
    
    def test_graph_retriever_pdf_document_full_text(
            self) \
            -> None:
        summarizer = \
            PDFSummarizer(
                pdf_path=self.pdf_path,
                openai_api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY)
    
        pdf_full_text = \
            summarizer.load_pdf_with_pdfplumber()
    
        graph_rag_orchestrator = \
            BoroGraphRagOrchestrator(
                data_set=pdf_full_text)
    
        graph_documents = \
            graph_rag_orchestrator.process_text(
                text=pdf_full_text,
                llm_transformer=graph_rag_orchestrator.llm_transformer)
    
        graph_rag_orchestrator.graph_documents = \
            graph_documents
    
        networkx_graph = \
            graph_rag_orchestrator.get_combined_networkx_graph_from_graph_documents()
    
        write_graphml(
            networkx_graph,
            self.output_file_path)
