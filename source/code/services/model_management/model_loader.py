from transformers import AutoTokenizer,AutoModelForCausalLM

def load_model(model_name, model_path):
    # Load the fine-tuned model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path+model_name+"_tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_path+model_name+"_model")
    return model, tokenizer