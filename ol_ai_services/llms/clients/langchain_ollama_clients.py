from langchain_community.llms import Ollama
from langchain.schema.messages import HumanMessage
from langchain.schema import StrOutputParser

from configurations.ol_configurations.nf_ollama_configurations import (
    NfOllamaConfigurations,
)


class LangChainOllamaClients:
    """
    Client for interacting with Ollama via LangChain.
    """
    def __init__(
            self,
            model=NfOllamaConfigurations.OLLAMA_MODEL,
            temperature=NfOllamaConfigurations.OLLAMA_TEMPERATURE,
            base_url=NfOllamaConfigurations.OLLAMA_BASE_URL,
            ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        
        # Initialize the Ollama LangChain client
        self.client = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            num_ctx=4096,  # Context window size
            top_k=NfOllamaConfigurations.OLLAMA_TOP_K,
            top_p=NfOllamaConfigurations.OLLAMA_TOP_P,
            repeat_penalty=NfOllamaConfigurations.OLLAMA_REPEAT_PENALTY,
            num_predict=NfOllamaConfigurations.OLLAMA_MAX_TOKENS,
        )
    
    def get_response(
            self,
            prompt: str,
            max_tokens: int = NfOllamaConfigurations.OLLAMA_MAX_TOKENS
            ):
        """
        Generate a text response using the Ollama API via LangChain.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Update max tokens if different from initialization
            if max_tokens != NfOllamaConfigurations.OLLAMA_MAX_TOKENS:
                self.client.num_predict = max_tokens
            
            # Process the prompt and get response
            response = self.client.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error during Ollama API call: {e}")
            return None
    
    def set_model(
            self,
            model: str):
        """
        Change the model used by the client.
        
        Args:
            model: The model name to use with Ollama
        """
        self.model = model
        # Recreate the client with the updated model
        self.client = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            num_ctx=4096,
            top_k=NfOllamaConfigurations.OLLAMA_TOP_K,
            top_p=NfOllamaConfigurations.OLLAMA_TOP_P,
            repeat_penalty=NfOllamaConfigurations.OLLAMA_REPEAT_PENALTY,
            num_predict=NfOllamaConfigurations.OLLAMA_MAX_TOKENS,
        )
    
    def set_temperature(
            self,
            temperature):
        """
        Change the temperature parameter for text generation.
        
        Args:
            temperature: Temperature value controlling randomness
        """
        self.temperature = temperature
        # Recreate the client with the updated temperature
        self.client = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            num_ctx=4096,
            top_k=NfOllamaConfigurations.OLLAMA_TOP_K,
            top_p=NfOllamaConfigurations.OLLAMA_TOP_P,
            repeat_penalty=NfOllamaConfigurations.OLLAMA_REPEAT_PENALTY,
            num_predict=NfOllamaConfigurations.OLLAMA_MAX_TOKENS,
        )
    
    def set_base_url(
            self,
            base_url: str):
        """
        Change the base URL for the Ollama API.
        
        Args:
            base_url: The base URL for the Ollama API
        """
        self.base_url = base_url
        # Recreate the client with the updated base URL
        self.client = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            num_ctx=4096,
            top_k=NfOllamaConfigurations.OLLAMA_TOP_K,
            top_p=NfOllamaConfigurations.OLLAMA_TOP_P,
            repeat_penalty=NfOllamaConfigurations.OLLAMA_REPEAT_PENALTY,
            num_predict=NfOllamaConfigurations.OLLAMA_MAX_TOKENS,
        )