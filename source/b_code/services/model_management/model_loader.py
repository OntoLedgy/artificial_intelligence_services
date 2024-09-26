from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

@run_and_log_function
def load_model(
        model_name,
        model_path):
    # Load the fine-tuned model and tokenizer

    model_folder_path = os.path.join(
        model_path,
        model_name,
        "fine_tuned_model")

    tokeniser_folder_path = os.path.join(
        model_path,
        model_name,
        "fine_tuned_tokeniser")

    tokenizer = AutoTokenizer.from_pretrained(
        tokeniser_folder_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_folder_path)

    return model, tokenizer
