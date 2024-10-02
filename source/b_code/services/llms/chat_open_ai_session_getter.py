from langchain_community.chat_models import ChatOpenAI


def get_chat_open_ai_session(
        api_key: str,
        temperature: float,
        model_name: str) \
        -> ChatOpenAI:
    chat_open_ai_session = \
        ChatOpenAI(
            api_key=api_key,
            temperature=temperature,
            model_name=model_name)
    
    return \
        chat_open_ai_session
