from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from .client_factory import LlmClientFactory, LlmClientType
from configurations.ol_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.ol_configurations.nf_ollama_configurations import NfOllamaConfigurations

# Rate limiter for OpenAI API
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
    check_every_n_seconds=2,
    max_bucket_size=10  # Allow for small bursts of requests
)

# In-memory cache for LLM responses
set_llm_cache(InMemoryCache())


@run_and_log_function()
def get_langchain_llm_model(
        client_type: LlmClientType = LlmClientType.LANGCHAIN_OPENAI,
        api_key: str = None,
        model_name: str = None,
        temperature: float = None,
        base_url: str = None) -> BaseLLM:
    """
    Get a LangChain LLM model of the specified type with the given parameters.
    
    Args:
        client_type: Type of LLM client to create
        api_key: API key for OpenAI models (only for OpenAI client types)
        model_name: Name of the model to use
        temperature: Temperature parameter for generation
        base_url: Base URL for Ollama models (only for Ollama client types)
        
    Returns:
        A LangChain LLM model
    """
    kwargs = {}
    
    if client_type in [LlmClientType.OPENAI, LlmClientType.LANGCHAIN_OPENAI]:
        # For OpenAI client types
        if api_key:
            kwargs['api_key'] = api_key
        if model_name:
            kwargs['model'] = model_name
        if temperature is not None:
            kwargs['temperature'] = temperature
            
        # Get the LangChain OpenAI client
        if client_type == LlmClientType.LANGCHAIN_OPENAI:
            client = LlmClientFactory.create_client(LlmClientType.LANGCHAIN_OPENAI, **kwargs)
            
            # Add rate limiter to the ChatOpenAI model
            langchain_model = ChatOpenAI(
                api_key=client.api_key,
                model=client.model,
                temperature=client.temperature,
                rate_limiter=rate_limiter
            )
            return langchain_model
            
    elif client_type in [LlmClientType.OLLAMA, LlmClientType.LANGCHAIN_OLLAMA]:
        # For Ollama client types
        if model_name:
            kwargs['model'] = model_name
        if temperature is not None:
            kwargs['temperature'] = temperature
        if base_url:
            kwargs['base_url'] = base_url
            
        # Get the LangChain Ollama client
        if client_type == LlmClientType.LANGCHAIN_OLLAMA:
            client = LlmClientFactory.create_client(LlmClientType.LANGCHAIN_OLLAMA, **kwargs)
            
            # Create an Ollama model
            langchain_model = Ollama(
                base_url=client.base_url,
                model=client.model,
                temperature=client.temperature,
                num_ctx=4096,
                top_k=NfOllamaConfigurations.OLLAMA_TOP_K,
                top_p=NfOllamaConfigurations.OLLAMA_TOP_P,
                repeat_penalty=NfOllamaConfigurations.OLLAMA_REPEAT_PENALTY,
                num_predict=NfOllamaConfigurations.OLLAMA_MAX_TOKENS
            )
            return langchain_model
            
    raise ValueError(f"Unsupported client type for LangChain model: {client_type}")


# For backward compatibility
@run_and_log_function()
def get_langchain_open_ai_model(
        api_key: str = NfOpenAiConfigurations.OPEN_AI_API_KEY,
        temperature: float = NfOpenAiConfigurations.OPEN_AI_TEMPERATURE,
        model_name: str = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O) -> BaseChatModel:
    """
    Get a LangChain OpenAI Chat model with the given parameters.
    This function is maintained for backward compatibility.
    
    Args:
        api_key: OpenAI API key
        temperature: Temperature parameter for generation
        model_name: Name of the model to use
        
    Returns:
        A LangChain ChatOpenAI model
    """
    return get_langchain_llm_model(
        client_type=LlmClientType.LANGCHAIN_OPENAI,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )