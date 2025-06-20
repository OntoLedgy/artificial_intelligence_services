import pytest

from ol_ai_services.llms.clients.langchain_open_ai_clients import LangChainOpenAiClients
from ol_ai_services.llms.clients.open_ai_clients import OpenAiClients


@pytest.fixture(autouse=True)
def openai_client(
        open_ai_configuration):
    api_key = open_ai_configuration.get_config(
        section_name="open_ai_configuration", key="api_key"
    )

    # ensure ChatOpenAI picks up the key via environment
    import os
    os.environ['OPENAI_API_KEY'] = api_key
    openai_client = OpenAiClients(api_key=api_key)

    return openai_client

@pytest.fixture(autouse=True)
def langchain_openai_client(
        open_ai_configuration):
    api_key = open_ai_configuration.get_config(
        section_name="open_ai_configuration", key="api_key"
    )

    langchain_openai_client = LangChainOpenAiClients(
            api_key=api_key)

    return langchain_openai_client