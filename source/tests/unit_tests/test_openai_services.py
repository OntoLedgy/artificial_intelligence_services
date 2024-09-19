import pytest

from source.code.common_utilities.configurations import Configurations
from source.code.services.object_model.configurations.OpenAIConfigurations import OpenAIConfigurations
from source.code.services.object_model.clients.OpenAiClient import OpenAiClient


class TestOpenAiServices:
    @pytest.fixture
    def openai_service(self):
        configuration_file = r"source\tests\configuration.json"

        configuration_manager = Configurations(
            configuration_file)

        configuration_manager.validate_config(
            OpenAIConfigurations,
            "open_ai_configuration")

        api_key = configuration_manager.get_config(
            "api_key")

        openai_service = OpenAiClient(
            api_key=api_key)

        return openai_service

    def test_gpt_response(
            self,
            openai_service):
        prompt = "Explain the theory of relativity in simple terms."

        response = openai_service.get_response(prompt)

        print(response)

        assert response is not None
