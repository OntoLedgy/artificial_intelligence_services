from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import (
    run_and_log_function,
)
from transformers import pipeline

from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)


@run_and_log_function
def generate_text_using_pipeline(
    model, tokenizer, input_text="In this study, we explore the effects of"
) -> str:
    # Create a text generation pipeline with the fine-tuned model
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Use the model to generate text
    output = generator(
        input_text, max_length=200, num_return_sequences=1, truncation=True
    )

    generated_text = output[0]["generated_text"]

    print("output")
    print(generated_text)

    return generated_text


@run_and_log_function
def generate_text_using_model(
        model,
        tokenizer,
        input_text: str = "In this study, we explore the effects of"
        ) -> str:
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text with specified parameters
    output_ids = model.generate(
        input_ids,
        max_length=NfGeneralConfigurations.TEXT_GENERATION_MAX_LENGTH,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=NfGeneralConfigurations.TEXT_GENERATION_TEMPERATURE,
        do_sample=True,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

    return generated_text
