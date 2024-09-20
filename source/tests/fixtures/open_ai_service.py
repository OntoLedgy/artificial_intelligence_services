import pytest

from services.object_model.clients.open_ai_clients import OpenAiClients


@pytest.fixture(autouse=True)
def openai_service(configuration_manager):
    api_key = configuration_manager.get_config(
        "api_key")

    openai_service = OpenAiClients(
        api_key=api_key)

    return openai_service
