from pydantic import BaseModel


class OpenAIConfigurations(BaseModel):
    api_key: str
    openai_organisation: str
    openai_project: str
    openai_model: str
    max_tokens: int
    temperature: float
    top_p: int
    # frequency_penalty: float
    # presence_penalty: float
    # stop: list
