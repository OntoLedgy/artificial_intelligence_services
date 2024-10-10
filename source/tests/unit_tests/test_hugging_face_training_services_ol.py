import json
import pytest
import os

from transformers import AutoModelForCausalLM

from services.chunking.chunked_texts_getter import get_chunked_texts

from services.fine_tuning.model_fine_tuner import fine_tune_model
from services.llms.text_generators import (
    generate_text_using_pipeline,
    generate_text_using_model,
    )
from services.model_management.model_loader import load_model
from services.tokenisation.tokeniser import Tokeniser


class TestHuggingFaceFineTunedModel:
    
    @pytest.fixture(
            autouse=True)
    def setup(
            self):
        self.pdf_folder = r"./data/inputs/pdf"
        
        self.chunked_data_file_path = (
            "data/outputs/training_data/accounting_training_data.jsonl"
        )
        self.tokenised_data_file_path = (
            "data/outputs/tokenised_data/accounting_tokenised_data.jsonl"
        )
        self.models_folder_path = "data/outputs/models"
        self.model_name = "accounting"
        
        self.model_folder_path = os.path.join(
                self.models_folder_path,
                self.model_name,
                "fine_tuned_model"
                )
        
        self.tokeniser_folder_path = os.path.join(
                self.models_folder_path,
                self.model_name,
                "fine_tuned_tokeniser"
                )
        
        self.model_type = "gpt2"
        self.tokenizer = Tokeniser(
            model_name=self.model_type)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_type)
        self.model.resize_token_embeddings(
            len(
                self.tokenizer.tokenizer))
        self.prompt = "describe different data modelling methodologies"
    
    
    def test_data_preparation(
            self):
        chunked_data = get_chunked_texts(
                source_texts_folder_path=self.pdf_folder,
                chunked_texts_output_file_path=self.chunked_data_file_path,
                chunk_size=512)
        
        print(
            chunked_data)
        
        # TODO: needs to be added to data_export
        with open(
                self.chunked_data_file_path,
                "w") as f:
            for entry in chunked_data.chunked_texts:
                json.dump(
                    entry,
                    f)
                f.write(
                    "\n")
    
    
    def test_tokenisation(
            self):
        self.tokenizer.tokenize(
            self.chunked_data_file_path)
        
        self.tokenizer.print_tokenized_data(
            num_samples=8)
        
        self.tokenizer.save_tokenized_data_to_file(
                output_file=self.tokenised_data_file_path
                )
    
    
    def test_fine_tuning(
            self):
        
        tokenized_dataset = self.tokenizer.read_tokenized_data_from_file(
                self.tokenised_data_file_path
                )
        
        print(
                tokenized_dataset[0])
        
        fine_tune_model(
            tokenized_dataset,
            self.tokenizer.tokenizer,
            self.model)
        
        self.model.save_pretrained(
            save_directory=self.model_folder_path)
        
        self.tokenizer.tokenizer.save_pretrained(
                save_directory=self.tokeniser_folder_path
                )
    
    
    def test_text_generation(
            self):
        model, tokeniser = load_model(
                self.model_name,
                self.models_folder_path)
        
        generate_text_using_pipeline(
                model,
                tokeniser,
                self.prompt)
        
        generate_text_using_model(
                model,
                tokeniser,
                self.prompt)
