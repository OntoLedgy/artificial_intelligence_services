import pytest

from services.llms.clients.open_ai_client import OpenAiClient


@pytest.fixture(autouse=True)
def openai_service(
        open_ai_configuration):
    api_key = open_ai_configuration.get_config(
        section_name="open_ai_configuration", key="api_key"
    )

    openai_service = OpenAiClient(api_key=api_key)

    return openai_service
