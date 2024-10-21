from services.llms.text_generators import generate_text_using_model
from services.llms.text_generators import generate_text_using_pipeline
from services.model_management.model_loader import load_model


#TODO: wrap this in a model class.
def orchestrate_text_generation(
        model_path: str,
        model_name: str,
        prompt: str) -> dict:
    model, tokeniser = load_model(
        model_name=model_name,
        model_path=model_path)
    
    generated_text_using_pipeline = \
        generate_text_using_pipeline(
            model=model,
            tokenizer=tokeniser,
            input_text=prompt)
    
    generated_text_using_model = \
        generate_text_using_model(
            model=model,
            tokenizer=tokeniser,
            input_text=prompt)
    
    generated_texts_dictionary = {
        "generated_text_using_pipeline": generated_text_using_pipeline,
        "generated_text_using_model"   : generated_text_using_model,
        }
    
    return \
        generated_texts_dictionary
