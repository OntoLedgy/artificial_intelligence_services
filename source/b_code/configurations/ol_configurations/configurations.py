from pydantic import ValidationError
from dynaconf import Dynaconf


class Configurations:
    
    def __init__(
            self,
            configuration_file_path,
            model_class=None):
        
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
                    key) for key in settings_section
                }
            
            model_instance = model_class(
                **config_data)
            
            setattr(
                    self,
                    section_name.lower(),
                    model_instance
                    )  # Store the validated config
            
            print(
                    f"{model_class.__name__} Configuration Validated and Loaded: {model_instance}"
                    )
        
        except ValidationError as e:
            print(
                f"Configuration Validation Error in {model_class.__name__}: {e}")
        
        except AttributeError:
            print(
                f"Missing configuration section: {section_name}")
    
    
    def get_config(
            self,
            section_name,
            key = None,
            skip_validation=False):
        
        """
        Get a configuration section or a specific key from the section.
        If skip_validation is True, fetch values directly from Dynaconf without requiring validation.
        """
        if skip_validation:
            # Directly access from Dynaconf without validation
            section = getattr(
                self.settings,
                section_name.upper(),
                None)
        else:
            # Check if the section was validated and stored
            section = getattr(
                self,
                section_name.lower(),
                None)
        
        if section is None:
            print(
                f"Configuration section {section_name} not found")
            return None
        
        if key:
            return getattr(
                section,
                key.lower(),
                None)
        
        return section
    
    
    def set_config(
            self,
            section_name,
            **kwargs):
        """Updates the configuration section dynamically."""
        if hasattr(
                self,
                section_name.lower()):
            for key, value in kwargs.items():
                setattr(
                    getattr(
                        self,
                        section_name.lower()),
                    key,
                    value)
            print(
                    f"Updated {section_name} Configuration: {getattr(self, section_name.lower())}"
                    )
        else:
            print(
                f"No existing configuration to update for {section_name}")
