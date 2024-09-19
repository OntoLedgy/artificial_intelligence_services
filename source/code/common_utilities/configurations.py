from pydantic import BaseModel, ValidationError
from dynaconf import Dynaconf


class Configurations:
    def __init__(self,
                 configuration_file_path):
        self.settings = Dynaconf(
            settings_files=[configuration_file_path],
        )

    def validate_config(
            self,
            model_class,
            section_name):

        try:

            settings_section = getattr(
                self.settings,
                section_name.upper())

            config_data = {
                key.lower(): getattr(
                    settings_section,
                    key) for key in settings_section}

            model_instance = model_class(
                **config_data)

            setattr(
                self,
                section_name.lower(),
                model_instance)  # Store the validated config

            print(f"{model_class.__name__} Configuration Validated and Loaded: {model_instance}")

        except ValidationError as e:
            print(f"Configuration Validation Error in {model_class.__name__}: {e}")

        except AttributeError:
            print(f"Missing configuration section: {section_name}")

    def get_config(self, section_name) -> str:
        """Gets a configuration section."""
        return getattr(self, section_name.lower(), None)

    def set_config(self, section_name, **kwargs):
        """Updates the configuration section dynamically."""
        if hasattr(self, section_name.lower()):
            for key, value in kwargs.items():
                setattr(getattr(self, section_name.lower()), key, value)
            print(f"Updated {section_name} Configuration: {getattr(self, section_name.lower())}")
        else:
            print(f"No existing configuration to update for {section_name}")
