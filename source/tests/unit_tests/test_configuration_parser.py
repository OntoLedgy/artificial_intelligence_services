from configurations.ol_configurations.configurations import Configurations
from configurations.ol_configurations.open_ai_configurations import OpenAIConfigurations


class TestConfigurationParsers:
    
    def test_parse_configuration_without_model_validation(self,
                                                       test_data_configuration_file_absolute_path):

        configuration_manager = Configurations(
                test_data_configuration_file_absolute_path)


        print(configuration_manager.get_config(section_name="graph_rag_paths",
                                               key="test_pdf_file_name",
                                               skip_validation = True)
              )
    
    def test_parse_configuration_with_model_validation(self,
                                                       open_ai_configuration_file_absolute_path):

        configuration_manager = Configurations(open_ai_configuration_file_absolute_path)

        configuration_manager.validate_config(
            model_class=OpenAIConfigurations,
            section_name="open_ai_configuration"
        )

        print(configuration_manager.get_config(section_name="open_ai_configuration",
                                               key="api_key")
              )

    def test_parse_configuration_with_invalid_file(self):
        assert True

    def test_parse_configuration_with_invalid_json(self):
        assert True

    def test_parse_configuration_with_missing_fields(self):
        assert True

    def test_parse_configuration_with_invalid_fields(self):
        assert True
