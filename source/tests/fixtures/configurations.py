import pytest
from b_code.configurations.ol_configurations.configurations import Configurations
from b_code.configurations.ol_configurations.open_ai_configurations import (
    OpenAIConfigurations,
    )


@pytest.fixture(
        scope="session")
def configuration_manager():
    configuration_file = r"source\tests\configurations\open_ai_configuration.json"
    
    configuration_manager = Configurations(
        configuration_file)
    
    configuration_manager.validate_config(
            model_class=OpenAIConfigurations,
            section_name="open_ai_configuration"
            )
    
    return configuration_manager
