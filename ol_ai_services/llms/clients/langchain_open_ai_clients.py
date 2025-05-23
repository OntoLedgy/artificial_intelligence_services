from langchain.chat_models import ChatOpenAI

from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
    )


class LangChainOpenAiClients:
    
    def __init__(
            self,
            api_key,
            model = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O,
            temperature = NfOpenAiConfigurations.OPEN_AI_TEMPERATURE,
            ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        self.client = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                )
    
    
    def get_response(
            self,
            prompt: str,
            max_tokens: int = 150
            ):
        try:
            # ChatOpenAI takes a list of messages for a conversation
            response = self.client(
                    messages=[{
                        "role": "user",
                        "content": prompt
                        }],
                    max_tokens=max_tokens,
                    )
            
            return response.choices[0].message['content']  # Assuming response.choices[0] has the message dictionary
        except Exception as e:
            print(
                f"Error during API call: {e}")
            return None
    
    
    def set_model(
            self,
            model: str):
        self.model = model
        # Recreate the client with the updated model
        self.client = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                )
    
    
    def set_temperature(
            self,
            temperature):
        self.temperature = temperature
        # Recreate the client with the updated temperature
        self.client = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                )
    
    
    def set_api_key(
            self,
            api_key: str):
        self.api_key = api_key
        # Recreate the client with the updated API key
        self.client = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                )
