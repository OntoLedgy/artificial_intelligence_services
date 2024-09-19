import pytest
from source.code.common_utilities.configurations import Configurations
from source.code.services.object_model.configurations.OpenAIConfigurations import OpenAIConfigurations


@pytest.fixture(scope="session")
def configuration_manager():

    configuration_file = r"source\tests\configurations\configuration.json"

    configuration_manager = Configurations(
        configuration_file)

    configuration_manager.validate_config(
        model_class=OpenAIConfigurations,
        section_name="open_ai_configuration")

    return configuration_manager
