from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
    check_every_n_seconds=2,
    max_bucket_size=10  # Allow for small bursts of requests
)

@run_and_log_function()
def get_langchain_open_ai_model(
        api_key: str,
        temperature: float,
        model_name: str) \
        -> ChatOpenAI:
    
    set_llm_cache(
            InMemoryCache())
    
    chat_open_ai_session = \
        ChatOpenAI(
            api_key=api_key,
            temperature=temperature,
            model_name=model_name,
            rate_limiter=rate_limiter)
    
    return \
        chat_open_ai_session
