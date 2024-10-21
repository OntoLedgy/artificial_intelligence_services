import os
import time
import numpy as np


from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from openai import RateLimitError

from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
)
from configurations.constants import UTF_8_ENCODING
from configurations.constants import WRITE_ACRONYM


# Initialize rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
    check_every_n_seconds=2,
    max_bucket_size=10  # Allow for small bursts of requests
)

@run_and_log_function()
def retrieve_similar_documents(
    query,
    model,
    index,
    documents,
    top_k=5,
    output_file="retrieved_similar_articles.txt",
):
    # Create an embedding for the query
    query_embedding = model.encode(
            [query],
            convert_to_tensor=False)

    # Search for similar articles in the index
    distances, indices = index.search(
            np.array(
                    query_embedding),
            top_k)

    # Retrieve the top-k articles
    retrieved_documents = [documents[i] for i in indices[0]]
    
    document_delimiter = "\n---\n"
    
    #TODO: make this a method in the exporter service
    with open(
            output_file,
            WRITE_ACRONYM,
            encoding=UTF_8_ENCODING) as file:
        for document in retrieved_documents:
            file.write(
                document + document_delimiter
            )

    return retrieved_documents

def truncate_context(
        context,
        max_tokens=NfGeneralConfigurations.DEFAULT_TRUNCATE_CONTEXT_MAX_TOKENS
):
    truncated_context_by_max_tokens = context[:max_tokens]
    return truncated_context_by_max_tokens


# Function to get response using retrieved documents
def get_response_using_retrieved_documents(
    query,
    model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_3_5_TURBO,
    input_file="retrieved_articles.txt",
    max_context_tokens=NfOpenAiConfigurations.DEFAULT_MAX_TRUNCATE_CONTEXT_TOKENS,
    retries = 3,
    # Number of retries before giving up
    backoff_factor = 2
    # Exponential backoff factor
):
    
    client = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=model_name,
            rate_limiter=rate_limiter
            )
    
    # Read the retrieved articles from the file
    with open(input_file, "r", encoding=UTF_8_ENCODING) as file:
        context = file.read()

    # Truncate the context to fit within the max context tokens
    truncated_context = truncate_context(
            context,
            max_context_tokens)

    # Craft the prompt with query and context
    prompt = (
        f"Context: {truncated_context}\n\n"
        f"Based on the provided context, please answer the following question: "
        f"{query}\n"
        f"Answer:"
    )
    
    for attempt in range(
            retries):
        try:
            response = client(
                    [
                        {
                            "role"   : "system",
                            "content": "You are a helpful assistant."
                            },
                        {
                            "role"   : "user",
                            "content": prompt
                            },
                        ])
            return response.content
        
        except RateLimitError as e:
            if attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                print(
                    f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(
                    wait_time)
            else:
                print(
                    f"Exceeded retry attempts. Error: {e}")
                raise

    return response.choices[0].message.content
