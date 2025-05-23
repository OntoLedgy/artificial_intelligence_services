from transformers import AutoModelForCausalLM
from fine_tuning.model_fine_tuner import fine_tune_model
from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
)
from llms.text_generators import generate_text_using_model
from llms.text_generators import generate_text_using_pipeline
from chunking.chunked_texts_getter import get_chunked_texts
from tokenisation.tokeniser import Tokeniser


# TODO: Modify the code to take the chunked data rather than exporting it and importing it from file
def orchestrate_text_generation_from_model_training_pipeline(
        source_texts_folder_path: str,
        output_folder_path: str,
        chunked_texts_output_file_path: str,
        prompt: str):
    chunked_texts = \
        get_chunked_texts(
            source_texts_folder_path=source_texts_folder_path,
            chunked_texts_output_file_path=chunked_texts_output_file_path)
    
    # TODO: To be used only for staged testing
    test = 'test'

    if test == 'test':
        return dict()

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=NfOpenAiConfigurations.OPEN_AI_MODEL_TYPE_NAME_GPT2
    )

    tokenizer = __initialise_model_and_tokeniser(
        model_type=NfOpenAiConfigurations.OPEN_AI_MODEL_TYPE_NAME_GPT2,
        pretrained_model=pretrained_model,
    )

    # TODO: modify inside code to tokenize the chunked data list rather than having to load it from file
    tokenised_dataset = __tokenise_dataset(
        tokenizer=tokenizer,
        chunked_data_file_path=chunked_texts_output_file_path
    )

    fine_tune_model(
        tokenized_dataset=tokenised_dataset,
        tokenizer=tokenizer,
        model=pretrained_model,
        output_path=output_folder_path + "/results",
        logging_dir=output_folder_path + "/logs",
    )

    generated_texts_dictionary = __generate_text_from_pretrained_model(
        pretrained_model=pretrained_model, tokenizer=tokenizer, prompt=prompt
    )

    return generated_texts_dictionary


def __initialise_model_and_tokeniser(model_type, pretrained_model) -> Tokeniser:
    tokenizer = Tokeniser(model_name=model_type)

    pretrained_model.resize_token_embeddings(len(tokenizer.tokenizer))

    return tokenizer


def __tokenise_dataset(tokenizer: Tokeniser, chunked_data_file_path: str):
    tokenizer.tokenize(data_files=chunked_data_file_path)

    tokenised_dataset = tokenizer.tokenized_dataset

    return tokenised_dataset


#TODO: this should be call the tuner model text generation method.
#TODO: consider if the orchestrator should do this or stop at delivering a model
def __generate_text_from_pretrained_model(
            pretrained_model,
            tokenizer: Tokeniser,
            prompt: str
        ) -> dict:
    generated_text_using_pipeline = generate_text_using_pipeline(
        model=pretrained_model,
        tokenizer=tokenizer,
        input_text=prompt
    )

    generated_text_using_model = generate_text_using_model(
        model=pretrained_model,
        tokenizer=tokenizer,
        input_text=prompt
    )

    generated_texts_dictionary = {
        "generated_text_using_pipeline": generated_text_using_pipeline,
        "generated_text_using_model": generated_text_using_model,
    }

    return generated_texts_dictionary
