from abc import ABC, \
    abstractmethod


class AbstractOpenAiClient(
        ABC):
    
    def __init__(
            self,
            api_key,
            model,
            temperature):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    
    @abstractmethod
    def get_response(
            self,
            prompt: str,
            max_tokens: int = 150):
        pass
    
    
    @abstractmethod
    def set_model(
            self,
            model: str):
        pass
    
    
    @abstractmethod
    def set_temperature(
            self,
            temperature):
        pass
    
    
    @abstractmethod
    def set_api_key(
            self,
            api_key: str):
        pass
