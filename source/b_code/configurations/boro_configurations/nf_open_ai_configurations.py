class NfOpenAiConfigurations:
    default_string_empty = \
        str()

    OPEN_AI_API_KEY = \
        default_string_empty

    OPEN_AI_ORGANISATION_KEY = \
        default_string_empty

    OPEN_AI_PROJECT_KEY = \
        default_string_empty

    OPEN_AI_MODEL_NAME_1 = \
        'gpt-4o-mini'
    
    OPEN_AI_MODEL_NAME_2 = \
        'gpt-3.5-turbo'
    
    OPEN_AI_MODEL_NAME_3 = \
        'gpt-4o'

    OPEN_AI_TEMPERATURE = \
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
