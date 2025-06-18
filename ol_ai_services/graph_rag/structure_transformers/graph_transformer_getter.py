from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(
        InMemoryCache())
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function

from configurations.ol_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from llms.chat_open_ai_session_getter import get_langchain_open_ai_model


@run_and_log_function()
def get_llm_graph_transformer(
        model_name: str,
        temperature: float) \
        -> LLMGraphTransformer:
    
    chat_open_ai_session = \
        get_langchain_open_ai_model(
                api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY,
                temperature=temperature,
                model_name=model_name)
    
    llm_graph_transformer = \
        LLMGraphTransformer(
                llm=chat_open_ai_session)
    
    return \
        llm_graph_transformer