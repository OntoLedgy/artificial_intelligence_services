import openai

from configurations.ol_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
)
from ol_ai_services.model_management.model_types import ModelTypes
from ol_ai_services.llms.llm_clients import AbstractOpenAiClient


class OpenAiClients(AbstractOpenAiClient):
    def __init__(
        self,
        api_key,
        model=ModelTypes.OPEN_AI_MODEL_NAME_GPT_4O,
        temperature=NfOpenAiConfigurations.OPEN_AI_TEMPERATURE,
    ):
        super().__init__(
            api_key,
            model,
            temperature)
        
        openai.api_key = self.api_key
        self.client = openai.ChatCompletion

    def get_response(
            self,
            prompt: str,
            max_tokens: int = 150
            ):
        
        try:
            response = self.client.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message
        except Exception as e:
            print(f"Error during API call: {e}")
            return None

    def set_model(
            self,
            model: str):
        
        self.model = model

    def set_temperature(
            self,
            temperature):
        self.temperature = temperature

    def set_api_key(
            self,
            api_key: str):
        
        self.api_key = api_key
        openai.api_key = self.api_key
