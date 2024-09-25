import os


class NfOpenAiConfigurations:
    default_string_empty = \
        str()

    OPEN_AI_API_KEY = \
        os.getenv(
            'OPENAI_API_KEY')

    OPEN_AI_ORGANISATION_KEY = \
        default_string_empty

    OPEN_AI_PROJECT_KEY = \
        default_string_empty

    # TODO: Are these constants rather than configurations?
    OPEN_AI_MODEL_NAME_GPT_4O_MINI = \
        'gpt-4o-mini'
    
    OPEN_AI_MODEL_NAME_GPT_3_5_TURBO = \
        'gpt-3.5-turbo'
    
    OPEN_AI_MODEL_NAME_GPT2 = \
        'gpt2'
    
    OPEN_AI_MODEL_NAME_GPT_4O = \
        'gpt-4o'

    OPEN_AI_TEMPERATURE = \
        0.7
    
    DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE = \
        0

    OPEN_AI_MAX_TOKENS = \
        1000
    
    DEFAULT_MAX_TRUNCATE_CONTEXT_TOKENS = \
        12000
    
    
    # def get_response(
    #         query,
    #         client,
    #         model_name = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_2,
    #         input_file = 'retrieved_articles.txt',
    #         max_context_tokens = 12000):
