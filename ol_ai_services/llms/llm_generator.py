from typing import Optional
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import (
    run_and_log_function,
)

from .client_factory import LlmClientFactory, LlmClientType
from .llm_clients import AbstractLlmClient
from configurations.ol_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)


class LlmGenerator:
    """
    A text generator class that uses LLM clients to generate text.
    This class demonstrates how different LLM clients can be swapped in and out.
    """
    
    def __init__(self, client_type: LlmClientType = LlmClientType.LANGCHAIN_OPENAI, **kwargs):
        """
        Initialize the text generator with a specific LLM client.
        
        Args:
            client_type: Type of LLM client to use
            **kwargs: Additional parameters for the client constructor
        """
        self.client = LlmClientFactory.create_client(client_type, **kwargs)
    
    def set_client(self, client_type: LlmClientType, **kwargs):
        """
        Change the LLM client used by this generator.
        
        Args:
            client_type: Type of LLM client to use
            **kwargs: Additional parameters for the client constructor
        """
        self.client = LlmClientFactory.create_client(client_type, **kwargs)
    
    @run_and_log_function()
    def generate_text(
            self,
            prompt: str,
            max_tokens: int = NfGeneralConfigurations.TEXT_GENERATION_MAX_LENGTH,
            temperature: Optional[float] = None
            ) -> str:
        """
        Generate text using the configured LLM client.
        
        Args:
            prompt: The input text prompt
            max_tokens: Maximum length of the generated response
            temperature: Override the client's temperature setting
            
        Returns:
            Generated text response
        """
        if temperature is not None:
            self.client.set_temperature(temperature)
            
        return self.client.get_response(prompt, max_tokens)