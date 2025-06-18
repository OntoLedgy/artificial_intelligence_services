from enum import Enum, auto
from typing import Optional, Dict, Any, Union

from .llm_clients import AbstractLlmClient
from .clients.open_ai_clients import OpenAiClients
from .clients.langchain_open_ai_clients import LangChainOpenAiClients
from .clients.ollama_clients import OllamaClient
from .clients.langchain_ollama_clients import LangChainOllamaClients

from configurations.ol_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.ol_configurations.nf_ollama_configurations import NfOllamaConfigurations


class LlmClientType(Enum):
    """
    Enum defining the types of LLM clients available.
    """
    OPENAI = auto()
    LANGCHAIN_OPENAI = auto()
    OLLAMA = auto()
    LANGCHAIN_OLLAMA = auto()


class LlmClientFactory:
    """
    Factory class for creating LLM clients based on the specified type.
    """
    
    @staticmethod
    def create_client(
            client_type: LlmClientType,
            **kwargs) -> AbstractLlmClient:
        """
        Create an LLM client of the specified type with the given parameters.
        
        Args:
            client_type: Type of LLM client to create
            **kwargs: Additional parameters for the client constructor
            
        Returns:
            An instance of the requested LLM client
            
        Raises:
            ValueError: If an unknown client type is specified
        """
        if client_type == LlmClientType.OPENAI:
            api_key = kwargs.get('api_key', NfOpenAiConfigurations.OPEN_AI_API_KEY)
            model = kwargs.get('model', NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O)
            temperature = kwargs.get('temperature', NfOpenAiConfigurations.OPEN_AI_TEMPERATURE)
            return OpenAiClients(api_key=api_key, model=model, temperature=temperature)
            
        elif client_type == LlmClientType.LANGCHAIN_OPENAI:
            api_key = kwargs.get('api_key', NfOpenAiConfigurations.OPEN_AI_API_KEY)
            model = kwargs.get('model', NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O)
            temperature = kwargs.get('temperature', NfOpenAiConfigurations.OPEN_AI_TEMPERATURE)
            return LangChainOpenAiClients(api_key=api_key, model=model, temperature=temperature)
            
        elif client_type == LlmClientType.OLLAMA:
            model = kwargs.get('model', NfOllamaConfigurations.OLLAMA_MODEL)
            temperature = kwargs.get('temperature', NfOllamaConfigurations.OLLAMA_TEMPERATURE)
            base_url = kwargs.get('base_url', NfOllamaConfigurations.OLLAMA_BASE_URL)
            return OllamaClient(model=model, temperature=temperature, base_url=base_url)
            
        elif client_type == LlmClientType.LANGCHAIN_OLLAMA:
            model = kwargs.get('model', NfOllamaConfigurations.OLLAMA_MODEL)
            temperature = kwargs.get('temperature', NfOllamaConfigurations.OLLAMA_TEMPERATURE)
            base_url = kwargs.get('base_url', NfOllamaConfigurations.OLLAMA_BASE_URL)
            return LangChainOllamaClients(model=model, temperature=temperature, base_url=base_url)
            
        else:
            raise ValueError(f"Unknown client type: {client_type}")