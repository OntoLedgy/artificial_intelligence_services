import numpy as np

from configurations.boro_configurations.nf_general_configurations import NfGeneralConfigurations
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.constants import UTF_8_ENCODING


def retrieve_similar_documents(
        query,
        model,
        index,
        documents,
        top_k = 5,
        output_file = "retrieved_similar_articles.txt"):
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
    
    with open(
            output_file,
            'w',
            encoding='utf-8') as file:
        for document in retrieved_documents:
            file.write(
                document + "\n---\n")  # Use "---" as a delimiter between articles
    
    return retrieved_documents


# Helper function to truncate context to fit within a token limit
def truncate_context(
        context,
        max_tokens = NfGeneralConfigurations.DEFAULT_TRUNCATE_CONTEXT_MAX_TOKENS):
    # Truncate the context to the specified number of characters (approximation for tokens)
    return context[:max_tokens]


# Function to get response using retrieved documents
def get_response(
        query,
        client,
        model_name = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_3_5_TURBO,
        input_file = 'retrieved_articles.txt',
        max_context_tokens = NfOpenAiConfigurations.DEFAULT_MAX_TRUNCATE_CONTEXT_TOKENS):

    # Read the retrieved articles from the file
    with open(
            input_file,
            'r',
            encoding=UTF_8_ENCODING) as file:
        context = file.read()
    
    # Truncate the context to fit within the max context tokens
    truncated_context = truncate_context(
            context,
            max_context_tokens)
    
    # Craft the prompt with query and context
    prompt = (f"Context: {truncated_context}\n\n"
              f"Based on the provided context, please answer the following question: "
              f"{query}\n"
              f"Answer:")
    
    # Use the OpenAI client to generate a response
    response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role"   : "system",
                    "content": "You are a helpful assistant."
                    },
                
                {
                    "role"   : "user",
                    "content": prompt
                    }
                ]
            )
    
    return response.choices[0].message.content
