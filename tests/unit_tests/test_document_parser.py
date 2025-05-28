import pytest
import os

from text_extraction.document_lexical_structure_extractor import DocumentParser, print_document_structure


class TestDocumentParsing():
    
    @pytest.fixture(
            autouse=True)
    def setup_method(
            self,
            inputs_folder_absolute_path):
        self.pdf_path = os.path.join(
                inputs_folder_absolute_path,
                r"D:\S\python\artificial_intelligence_services\tests\data\inputs\pdf\accounting\Blums_Conceptualizing-resources.pdf"
                )
        
    def test_parse_document(self):
        parser = DocumentParser()
        doc_structure = parser.parse(
                r"D:\S\python\artificial_intelligence_services\tests\data\inputs\pdf\accounting\Blums_Conceptualizing-resources.pdf",
                )
        print_document_structure(doc_structure)