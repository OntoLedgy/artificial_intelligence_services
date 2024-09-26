import json
from transformers import AutoModelForCausalLM
from services.fine_tuning.model_fine_tuner import train_model
from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from services.data_preparation.pdf_services import extract_text_from_pdfs
from services.data_preparation.prepare_data import prepare_data_for_training
from services.llms.text_generators import generate_text_using_model
from services.llms.text_generators import generate_text_using_pipeline
from services.tokenisation.tokeniser import Tokeniser


# TODO: Modify the code to take the chunked data rather than exporting it and importing it from file
def orchestrate_fine_tuning_hugging_face(
        pdf_folder_path: str,
        chunked_data_file_path: str,
        prompt: str):
    chunked_data = \
        __prepare_data_for_model_training(
                pdf_folder_path=pdf_folder_path,
                chunked_data_file_path=chunked_data_file_path)
    
    pretrained_model = \
        AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=NfOpenAiConfigurations.OPEN_AI_MODEL_TYPE_NAME_GPT2)
    
    tokenizer = \
        __initialise_model_and_tokeniser(
                model_type=NfOpenAiConfigurations.OPEN_AI_MODEL_TYPE_NAME_GPT2,
                pretrained_model=pretrained_model)
    
    # TODO: modify inside code to tokenize the chunked data list rather than having to load it from file
    tokenised_dataset = \
        __tokenise_dataset(
                tokenizer=tokenizer,
                chunked_data_file_path=chunked_data_file_path)
    
    train_model(
            tokenized_dataset=tokenised_dataset,
            tokenizer=tokenizer,
            model=pretrained_model)
    
    generated_texts_dictionary = \
        __generate_text_from_pretrained_model(
                pretrained_model=pretrained_model,
                tokenizer=tokenizer,
                prompt=prompt)
    
    return \
        generated_texts_dictionary
    
    
def __prepare_data_for_model_training(
        pdf_folder_path: str,
        chunked_data_file_path: str) \
        -> list:
    pdf_texts = \
        extract_text_from_pdfs(
                pdf_folder=pdf_folder_path)
    
    chunked_data = \
        prepare_data_for_training(
                texts=pdf_texts,
                chunk_size=NfGeneralConfigurations.DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING)
    
    write_list_of_dictionaries_to_json_file(
            output_file_path=chunked_data_file_path,
            list_of_dictionaries=chunked_data)
    
    return \
        chunked_data
    
    
def __initialise_model_and_tokeniser(
        model_type,
        pretrained_model) \
        -> Tokeniser:
    tokenizer = \
        Tokeniser(
                model_name=model_type)
    
    pretrained_model.resize_token_embeddings(
            len(
                    tokenizer.tokenizer))
    
    return \
        tokenizer


def __tokenise_dataset(
        tokenizer: Tokeniser,
        chunked_data_file_path: str):
    tokenizer.tokenize(
            chunked_data_file_path=chunked_data_file_path)
    
    tokenised_dataset = \
        tokenizer.tokenized_dataset
    
    return \
        tokenised_dataset


def __generate_text_from_pretrained_model(
        pretrained_model,
        tokenizer: Tokeniser,
        prompt: str) \
        -> dict:
    generated_text_using_pipeline = \
        generate_text_using_pipeline(
                model=pretrained_model,
                tokenizer=tokenizer,
                input_text=prompt)
        
    generated_text_using_model = \
        generate_text_using_model(
                model=pretrained_model,
                tokenizer=tokenizer,
                input_text=prompt)
    
    generated_texts_dictionary = {
            'generated_text_using_pipeline': generated_text_using_pipeline,
            'generated_text_using_model': generated_text_using_model
        }
    
    return \
        generated_texts_dictionary


def write_list_of_dictionaries_to_json_file(
        output_file_path: str,
        list_of_dictionaries: list) \
        -> None:
    with open(
            output_file_path,
            'w') as output_file:
        for entry in list_of_dictionaries:
            json.dump(
                    entry,
                    output_file)
            output_file.write(
                    '\n')
