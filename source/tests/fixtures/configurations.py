import pytest
from configurations.configurations import Configurations
from configurations.open_ai_configurations import OpenAIConfigurations


@pytest.fixture(scope="session")
def configuration_manager():

    configuration_file = r"source\tests\configurations\configuration.json"

    configuration_manager = Configurations(
        configuration_file)

    configuration_manager.validate_config(
        model_class=OpenAIConfigurations,
        section_name="open_ai_configuration")

    return configuration_manager
