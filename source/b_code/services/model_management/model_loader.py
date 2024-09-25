from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from transformers import AutoTokenizer, AutoModelForCausalLM


@run_and_log_function
def load_model(
        model_name,
        model_path):
    # Load the fine-tuned model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path + model_name + "_tokenizer")

    model = AutoModelForCausalLM.from_pretrained(
        model_path + model_name + "_model")

    return model, tokenizer
