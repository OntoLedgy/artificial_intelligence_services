from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokeniser(
        model_name,
        model_path):
    tokeniser = AutoTokenizer.from_pretrained(model_path + model_name + "_tokenizer")

    model = AutoModelForCausalLM.from_pretrained(model_path + model_name + "_model")

    return model, tokeniser
