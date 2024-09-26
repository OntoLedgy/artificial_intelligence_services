from services.llms.text_generators import generate_text_using_model
from services.llms.text_generators import generate_text_using_pipeline
from services.model_management.model_loader import load_model


def orchestrate_text_generation(
        model_path: str,
        model_name: str,
        prompt: str):
    model, tokeniser = \
        load_model(
            model_name=model_name,
            model_path=model_path)
    
    generate_text_using_pipeline(
            model=model,
            tokenizer=tokeniser,
            input_text=prompt)
    
    generate_text_using_model(
            model=model,
            tokenizer=tokeniser,
            input_text=prompt)