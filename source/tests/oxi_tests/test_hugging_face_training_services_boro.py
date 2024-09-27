import json
import os
import pytest
from nf_common_source.code.services.reporting_service.reporters.log_file import LogFiles
from transformers import AutoModelForCausalLM
from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from services.data_preparation.pdf_services import extract_text_from_pdfs
from services.data_preparation.prepare_data import prepare_data_for_training
from services.fine_tuning.model_fine_tuner import train_model
from services.orchestrators.model_pdf_data_preparer import prepare_model_pdf_data
from services.orchestrators.text_generation_from_model_training_pipeline_orchestrator import orchestrate_text_generation_from_model_training_pipeline
from services.orchestrators.text_generation_orchestrator import orchestrate_text_generation
from services.tokenisation.tokeniser import Tokeniser
from z_sandpit.test_data.configuration.z_sandpit_test_constants import Z_SANDPIT_TEST_DATA_FOLDER_PATH


class TestHuggingFaceFineTunedModelBoro:
    
    @pytest.fixture(
            autouse=True)
    def setup(
            self):
        self.pdf_folder = \
            os.path.join(
                    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
                    'inputs')
        
        self.z_sandpit_outputs_folder = \
            os.path.join(
                    Z_SANDPIT_TEST_DATA_FOLDER_PATH,
                    'outputs')
        
        self.models_folder_path = \
            os.path.join(
                    self.z_sandpit_outputs_folder,
                    'models')
        
        self.model_folder_path = os.path.join(
                self.models_folder_path,
                NfGeneralConfigurations.HUGGING_FACE_MODEL_NAME,
                "fine_tuned_model")
        
        self.tokeniser_folder_path = os.path.join(
                self.models_folder_path,
                NfGeneralConfigurations.HUGGING_FACE_MODEL_NAME,
                "fine_tuned_tokeniser")
        
        self.chunked_data_file_path = \
            os.path.join(
                    self.z_sandpit_outputs_folder,
                    'training_data',
                    'accounting_training_data.jsonl')
        
        self.tokenised_data_file_path = \
            os.path.join(
                    self.z_sandpit_outputs_folder,
                    'tokenised_data',
                    'accounting_tokenised_data.jsonl')
        self.pretrained_model_name_or_path = \
            os.path.join(
                    self.z_sandpit_outputs_folder,
                    'models',
                    'accounting_fine_tuned_model')
        
        self.pretrained_tokenizer_name_or_path = \
            os.path.join(
                    self.z_sandpit_outputs_folder,
                    'models',
                    'accounting_fine_tuned_tokenizer')
        
        self.prompt = \
            'what is BORO?'
        
        self.model_type = \
            NfOpenAiConfigurations.OPEN_AI_MODEL_TYPE_NAME_GPT2
        
        self.__initialise_model_and_tokeniser()
        
        LogFiles.open_log_file(
                folder_path=os.path.join(
                        self.z_sandpit_outputs_folder,
                        'logs'))
    
    def __initialise_model_and_tokeniser(
            self) \
            -> None:
        self.model = \
            AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_type)
        
        self.tokenizer = \
            Tokeniser(
                    model_name=self.model_type)
        
        self.model.resize_token_embeddings(
                len(
                        self.tokenizer.tokenizer))
    
    def test_data_preparation(
            self):
        chunked_data = \
            prepare_model_pdf_data(
                    pdf_folder_path=self.pdf_folder,
                    chunked_data_file_path=self.chunked_data_file_path)
    
    def test_tokenisation(
            self):
        self.tokenizer.tokenize(
                self.chunked_data_file_path)
        
        self.tokenizer.print_tokenized_data(
                num_samples=8)
        
        self.tokenizer.save_tokenized_data_to_file(
                output_file=self.tokenised_data_file_path)
    
    def test_fine_tuning(
            self):
        tokenized_dataset = \
            self.tokenizer.read_tokenized_data_from_file(
                input_file=self.tokenised_data_file_path)
        
        print(
                tokenized_dataset[0])
        
        train_model(
                tokenized_dataset=tokenized_dataset,
                tokenizer=self.tokenizer.tokenizer,
                model=self.model,
                output_path=self.z_sandpit_outputs_folder + '/results',
                logging_dir=LogFiles.folder_path)
        
        self.model.save_pretrained(
                save_directory=self.model_folder_path)
        
        self.tokenizer.tokenizer.save_pretrained(
                save_directory=self.tokeniser_folder_path)
    
    
    def test_text_generation(
            self):
        model_name = \
            NfGeneralConfigurations.HUGGING_FACE_MODEL_NAME
        
        orchestrate_text_generation(
                self.models_folder_path,
                model_name,
                self.prompt)
    
    def test_text_generation_from_model_training_pipeline(
            self):
        generated_texts_dictionary = \
            orchestrate_text_generation_from_model_training_pipeline(
                    pdf_folder_path=self.pdf_folder,
                    output_folder_path=self.z_sandpit_outputs_folder,
                    chunked_data_file_path=self.chunked_data_file_path,
                    prompt=self.prompt)

        print(
                generated_texts_dictionary)
