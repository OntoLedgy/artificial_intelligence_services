import os


class NfOpenAiConfigurations:
    default_string_empty = str()

    OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

    OPEN_AI_ORGANISATION_KEY = default_string_empty

    OPEN_AI_PROJECT_KEY = default_string_empty

    OPEN_AI_TEMPERATURE = 0.7

    DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE = 0

    OPEN_AI_MAX_TOKENS = 1000

    DEFAULT_MAX_TRUNCATE_CONTEXT_TOKENS = 12000
