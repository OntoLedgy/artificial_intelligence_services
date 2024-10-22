import pytest
from test_unstructured.test_utils import output_jsonl_file

from services.chunking.chunked_texts_getter import get_chunked_texts
from services.text_extraction.pdf_document_extractor import extract_text_from_pdfs_in_folder

from services.llms.model_types import ModelTypes
from services.tokenisation.tokeniser_factories import TokeniserFactory
from services.tokenisation.tokeniser_types import TokeniserTypes


class TestTokenisationServices:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        pdf_folder = r"./data/inputs/pdf"

        self.pdf_text = extract_text_from_pdfs_in_folder(
            pdf_folder)

        self.chunked_file_name_and_path = "./data/outputs/chunked_data/chunked_data.json"
        self.tokenised_data_file_name_and_path = "./data/outputs/tokenised_data/tokenised_data.json"



    def test_tokenisation_hugging_face(self):

        huggingface_tokeniser = TokeniserFactory(
            tokeniser_type=TokeniserTypes.HUGGING_FACE,
            model_name=ModelTypes.OPEN_AI_MODEL_NAME_GPT2
        ).get_tokeniser()

        tokenised_data = huggingface_tokeniser.tokenize(
            self.chunked_file_name_and_path)

        print(
            tokenised_data)

        huggingface_tokeniser.save_tokenized_data_to_file(
            output_file=self.tokenised_data_file_name_and_path
        )

        assert tokenised_data is not None
        assert len(tokenised_data) > 0
