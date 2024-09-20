from openai import OpenAI
import openai


class OpenAiClient:
    def __init__(
            self,
            api_key,
            model="gpt-3.5-turbo",
            temperature=0.7):

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        openai.api_key = self.api_key

        self.client = OpenAI(
            api_key='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',
            # organization="org-JVbfEJAWpvfoAZKVfrF4aLLp",
            # project="proj_8vpS67K6GLSjLkJQlvVCgRGt"
        )

    def get_response(
            self,
            prompt: str,
            max_tokens: int = 150):

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message
        except Exception as e:
            print(f"Error during API call: {e}")
            return None

    def set_model(self, model: str):
        self.model = model

    def set_temperature(
            self,
            temperature):

        self.temperature=temperature

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key
