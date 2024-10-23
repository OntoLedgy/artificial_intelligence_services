import os
import pytest
from nf_common_source.code.services.reporting_service.reporters.log_file import LogFiles
from transformers import AutoModelForCausalLM
from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
)
from configurations.constants import TEXT_GENERATION_METHOD_COLUMN_NAME
from configurations.constants import TEXT_GENERATION_OUTPUT_COLUMN_NAME
from services.data_export.dictionary_of_strings_to_csv_exporter import export_dictionary_of_strings_to_csv
from services.fine_tuning.model_fine_tuner import fine_tune_model
from services.chunking.chunked_texts_getter import get_chunked_texts
from services.llms.orchestrators.text_generation_from_model_training_pipeline_orchestrator import (
    orchestrate_text_generation_from_model_training_pipeline,
)
from services.llms.orchestrators.text_generation_orchestrator import (
    orchestrate_text_generation,
)
from services.tokenisation.tokeniser import Tokeniser


class TestHuggingFaceFineTunedModelBoro:
    @pytest.fixture(autouse=True)
    def setup(self,
              test_data_folder_absolute_path,
              outputs_folder_absolute_path,
              inputs_folder_absolute_path):
        self.pdf_folder = os.path.join(
                inputs_folder_absolute_path,
                "pdf/accounting")

        self.test_outputs_folder = outputs_folder_absolute_path

        self.models_folder_path = os.path.join(
                self.test_outputs_folder,
                "models")

        self.model_folder_path = os.path.join(
            self.models_folder_path,
            NfGeneralConfigurations.HUGGING_FACE_MODEL_NAME,
            "fine_tuned_model",
        )

        self.tokeniser_folder_path = os.path.join(
            self.models_folder_path,
            NfGeneralConfigurations.HUGGING_FACE_MODEL_NAME,
            "fine_tuned_tokeniser",
        )

        self.chunked_data_file_path = os.path.join(
            self.test_outputs_folder,
            "training_data",
            "accounting_training_data.jsonl",
        )

        self.tokenised_data_file_path = os.path.join(
            self.test_outputs_folder,
            "tokenised_data",
            "accounting_tokenised_data.jsonl",
        )
        
        self.pretrained_model_name_or_path = os.path.join(
            self.test_outputs_folder,
                "models",
                "accounting_fine_tuned_model"
        )

        self.pretrained_tokenizer_name_or_path = os.path.join(
            self.test_outputs_folder,
                "models",
                "accounting_fine_tuned_tokenizer"
        )

        self.prompt = "what is bCLEARer?"

        self.model_type = NfOpenAiConfigurations.OPEN_AI_MODEL_TYPE_NAME_GPT2

        self.__initialise_model_and_tokeniser()
        
        LogFiles.open_log_file(
            folder_path=os.path.join(test_data_folder_absolute_path, "logs")
        )

    def __initialise_model_and_tokeniser(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_type
        )

        self.tokenizer = Tokeniser(
                model_name=self.model_type)

        self.model.resize_token_embeddings(
                len(
                        self.tokenizer.tokenizer
                        )
                )

    def test_data_preparation(self):
        chunked_texts = \
            get_chunked_texts(
                source_texts_folder_path=self.pdf_folder,
                chunked_texts_output_file_path=self.chunked_data_file_path)
        
        assert len(chunked_texts.texts.source_texts) > 0

    def test_tokenisation(self):
        self.tokenizer.tokenize(
                self.chunked_data_file_path)

        self.tokenizer.print_tokenized_data(
                num_samples=8)

        self.tokenizer.save_tokenized_data_to_file(
            output_file=self.tokenised_data_file_path)

    def test_fine_tuning(self):
        tokenized_dataset = self.tokenizer.read_tokenized_data_from_file(
            input_file=self.tokenised_data_file_path
        )

        print(tokenized_dataset[0])
        
        results_folder_path = os.path.join(
                self.test_outputs_folder,
                "results"
                )
        
        fine_tune_model(
            tokenized_dataset=tokenized_dataset,
            tokenizer=self.tokenizer.tokenizer,
            model=self.model,
            output_path=results_folder_path,
            logging_dir=LogFiles.folder_path,
        )

        self.model.save_pretrained(
                save_directory=self.model_folder_path)

        self.tokenizer.tokenizer.save_pretrained(
            save_directory=self.tokeniser_folder_path
        )

    def test_text_generation(self):
        model_name = NfGeneralConfigurations.HUGGING_FACE_MODEL_NAME

        generated_texts_dictionary = \
            orchestrate_text_generation(
                    self.models_folder_path,
                    model_name,
                    self.prompt
                    )
        
        output_file_path = os.path.join(self.test_outputs_folder,
                                        'generated_text/generated_text.csv')
                
        export_dictionary_of_strings_to_csv(
                output_file_path=output_file_path,
                dictionary_of_strings=generated_texts_dictionary,
                keys_column_name=TEXT_GENERATION_METHOD_COLUMN_NAME,
                values_column_name=TEXT_GENERATION_OUTPUT_COLUMN_NAME)
        
    def test_text_generation_from_model_training_pipeline(self):
        generated_texts_dictionary = (
            orchestrate_text_generation_from_model_training_pipeline(
                source_texts_folder_path=self.pdf_folder,
                output_folder_path=self.test_outputs_folder,
                chunked_texts_output_file_path=self.chunked_data_file_path,
                prompt=self.prompt,
            )
        )

        print(generated_texts_dictionary)
