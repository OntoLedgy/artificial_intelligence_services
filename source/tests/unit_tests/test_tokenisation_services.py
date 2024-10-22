import pytest
import os
from services.llms.model_types import ModelTypes
from services.text_extraction.pdf_folder_extractor import extract_text_from_pdfs_in_folder
from services.tokenisation.tokeniser_factories import TokeniserFactory
from services.tokenisation.tokeniser_types import TokeniserTypes


class TestTokenisationServices:

    @pytest.fixture(autouse=True)
    def setup_method(self,
                     inputs_folder_absolute_path):
        
        pdf_folder = os.path.join(
                inputs_folder_absolute_path,
                'pdf')

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
