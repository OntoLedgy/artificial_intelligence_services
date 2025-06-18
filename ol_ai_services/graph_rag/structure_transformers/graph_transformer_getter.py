from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(
        InMemoryCache())
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function

from configurations.ol_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.ol_configurations.nf_ollama_configurations import NfOllamaConfigurations
from ol_ai_services.llms.langchain_model_getter import get_langchain_llm_model
from ol_ai_services.llms.client_factory import LlmClientType


@run_and_log_function()
def get_llm_graph_transformer(
        model_name: str,
        temperature: float,
        client_type: LlmClientType = LlmClientType.LANGCHAIN_OPENAI,
        api_key: str = NfOpenAiConfigurations.OPEN_AI_API_KEY,
        base_url: str = NfOllamaConfigurations.OLLAMA_BASE_URL) \
        -> LLMGraphTransformer:
    """
    Get a LLMGraphTransformer using the specified LLM client type.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for generation
        client_type: Type of LLM client to use (default: LANGCHAIN_OPENAI)
        api_key: API key for OpenAI models (only used with OpenAI client types)
        base_url: Base URL for Ollama models (only used with Ollama client types)
        
    Returns:
        A LLMGraphTransformer configured with the specified LLM
    """
    
    # Get the appropriate LangChain model based on client type
    langchain_model = get_langchain_llm_model(
        client_type=client_type,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        base_url=base_url
    )
    
    # Create and return the graph transformer with the LLM
    llm_graph_transformer = LLMGraphTransformer(llm=langchain_model)
    
    return llm_graph_transformer