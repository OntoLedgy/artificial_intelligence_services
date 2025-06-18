from pydantic import BaseModel


class OllamaConfigurations(BaseModel):
    """
    Configuration schema for Ollama API.
    """
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.1
    timeout: int = 120