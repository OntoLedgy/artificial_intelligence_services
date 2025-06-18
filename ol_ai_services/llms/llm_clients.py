from abc import ABC, \
    abstractmethod


class AbstractLlmClient(ABC):
    """
    Generic abstract base class for all language model clients.
    This provides the common interface for all LLM clients regardless of their provider.
    """
    
    def __init__(
            self,
            model,
            temperature):
        self.model = model
        self.temperature = temperature
    
    @abstractmethod
    def get_response(
            self,
            prompt: str,
            max_tokens: int = 150):
        """
        Generate a text response to a prompt.
        
        Args:
            prompt: The input text prompt
            max_tokens: Maximum length of the generated response
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    def set_model(
            self,
            model: str):
        """
        Change the model used by this client.
        
        Args:
            model: The model identifier
        """
        pass
    
    @abstractmethod
    def set_temperature(
            self,
            temperature):
        """
        Change the temperature parameter for text generation.
        
        Args:
            temperature: Temperature value controlling randomness
        """
        pass


class AbstractOpenAiClient(AbstractLlmClient):
    """
    Abstract client specifically for OpenAI API compatible services.
    Extends the base LLM client with OpenAI-specific functionality.
    """
    
    def __init__(
            self,
            api_key,
            model,
            temperature):
        super().__init__(model, temperature)
        self.api_key = api_key
    
    @abstractmethod
    def set_api_key(
            self,
            api_key: str):
        """
        Change the API key used for authentication.
        
        Args:
            api_key: The API key for authentication
        """
        pass
