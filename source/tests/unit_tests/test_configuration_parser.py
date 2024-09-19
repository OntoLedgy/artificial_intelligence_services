from source.code.common_utilities.configurations import Configurations
from source.code.services.object_model.configurations.OpenAIConfigurations import OpenAIConfigurations


class TestConfigurationParsers:
    def test_parse_configuration(self
                                 ):
        configuration_file = r"source\tests\configurations\configuration.json"

        configuration_manager = Configurations(
            configuration_file)

        configuration_manager.validate_config(
            model_class=OpenAIConfigurations,
            section_name="open_ai_configuration")

        print(configuration_manager.get_config("api_key"))

    def test_parse_configuration_with_invalid_file(self):
        assert False

    def test_parse_configuration_with_invalid_json(self):
        assert False

    def test_parse_configuration_with_missing_fields(self):
        assert False

    def test_parse_configuration_with_invalid_fields(self):
        assert False