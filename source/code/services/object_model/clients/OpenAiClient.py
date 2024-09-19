from openai import OpenAI
import openai


class OpenAiClient:
    def __init__(
            self,
            api_key,
            model="gpt-4o",
            temperature=0.7):

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        openai.api_key = self.api_key

        self.client = OpenAI(
            api_key=self.api_key,
            organization="org-JVbfEJAWpvfoAZKVfrF4aLLp",
            project="proj_8vpS67K6GLSjLkJQlvVCgRGt"
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
