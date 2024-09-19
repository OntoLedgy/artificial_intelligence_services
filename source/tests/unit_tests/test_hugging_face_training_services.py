import json
import pytest

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from code.common_utilities.pdf_services import extract_text_from_pdfs
from code.services.data_preparation.prepare_data import prepare_data_for_training
from code.services.fine_tuning.model_fine_tuner import train_model
from code.services.llms.text_generators import generate_text_using_pipeline, generate_text_using_model
from code.services.model_management.model_loader import load_model
from code.services.tokenisation.tokeniser import Tokeniser


class TestHuggingFaceFineTunedModel:

    @pytest.fixture(autouse=True)
    def setup(self):
        pdf_folder = r'D:\OneDrives\OneDrive - OntoLedgy\Ontology of Money\Literature Review\Accounting'
        self.pdf_text = extract_text_from_pdfs(pdf_folder)
        self.chunked_data_file_path = 'data/outputs/training_data/accounting_training_data.jsonl'
        self.tokenised_data_file_path = 'data/outputs/tokenised_data/accounting_tokenised_data.jsonl'
        self.pretrained_model_name_or_path = 'data/outputs/models/accounting_fine_tuned_model'
        self.pretrained_tokenizer_name_or_path = 'data/outputs/models/accounting_fine_tuned_tokenizer'
        self.model_type = 'gpt2'
        self.tokenizer = Tokeniser(model_name=self.model_type)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_type)
        self.model.resize_token_embeddings(len(self.tokenizer.tokenizer))
        self.prompt = "give a history of accounting methods"

    def test_data_preparation(self):
        chunked_data = prepare_data_for_training(self.pdf_text, chunk_size=512)
        print(chunked_data)

        # Save the dataset in JSONL format
        with open(self.chunked_data_file_path, 'w') as f:
            for entry in chunked_data:
                json.dump(entry, f)
                f.write('\n')

    def test_tokenisation(self):
        self.tokenizer.tokenize(
            self.chunked_data_file_path)

        self.tokenizer.print_tokenized_data(
            num_samples=2)

        self.tokenizer.save_tokenized_data_to_file(
            output_file=self.tokenised_data_file_path)

    def test_fine_tuning(self):
        # Example usage
        tokenized_dataset = self.tokenizer.read_tokenized_data_from_file(
            self.tokenised_data_file_path)

        # To inspect the loaded data
        print(tokenized_dataset[0])

        train_model(tokenized_dataset,
                    self.tokenizer,
                    self.model)

        self.model.save_pretrained(
            self.pretrained_model_name_or_path)

        self.tokenizer.tokenizer.save_pretrained(
            self.pretrained_tokenizer_name_or_path)

    def test_text_generation(self):

        model_path = r'data/outputs/models/'
        model_name = "accounting_fine_tuned"

        model, tokeniser = load_model(
            model_name,
            model_path)

        generate_text_using_pipeline(
            model,
            tokeniser,
            self.prompt)

        generate_text_using_model(
            model,
            tokeniser,
            self.prompt)
