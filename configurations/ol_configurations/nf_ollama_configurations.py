import os


class NfOllamaConfigurations:
    """
    Default configurations for Ollama API.
    """
    default_string_empty = str()

    # Base URL for Ollama API, defaults to localhost
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Default model to use
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    # Generation parameters
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))
    
    # Additional Ollama-specific parameters
    OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "1.0"))
    OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "40"))
    OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
    
    # Request parameters
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))