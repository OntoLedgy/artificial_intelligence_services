import pytest
import os
import json
from configurations.ol_configurations.configurations import Configurations
from configurations.ol_configurations.open_ai_configurations import (
    OpenAIConfigurations,
)

@pytest.fixture(scope="session")
def open_ai_configuration_file_absolute_path(configurations_folder_absolute_path):
    open_ai_configuration_file = r"open_ai_configuration.json"
    
    open_ai_configuration_file_absolute_path = os.path.join(
            configurations_folder_absolute_path,
            open_ai_configuration_file)
    
    return open_ai_configuration_file_absolute_path


@pytest.fixture(scope="session")
def open_ai_configuration(open_ai_configuration_file_absolute_path):
    
    configuration_manager = Configurations(
            open_ai_configuration_file_absolute_path)

    configuration_manager.validate_config(
        model_class=OpenAIConfigurations, section_name="open_ai_configuration"
    )

    return configuration_manager

@pytest.fixture(scope="session")
def test_data_configuration_file_absolute_path(
        configurations_folder_absolute_path):
    data_configuration_file_name = r"test_data_configuration.json"
    
    data_configuration_file_absolute_path = os.path.join(
            configurations_folder_absolute_path,
            data_configuration_file_name)

    return data_configuration_file_absolute_path

@pytest.fixture(scope="session")
def test_data_configuration(test_data_configuration_file_absolute_path):
    
    configuration_manager = Configurations(
            test_data_configuration_file_absolute_path)
    
    return configuration_manager

@pytest.fixture(scope="session")
def graph_configuration_file_absolute_path(configurations_folder_absolute_path):
    graph_configuration_file = r"graph_configuration.json"
    
    graph_configuration_file_absolute_path = os.path.join(
            configurations_folder_absolute_path,
            graph_configuration_file)
    
    return graph_configuration_file_absolute_path

@pytest.fixture(scope="session")
def neo4j_config(graph_configuration_file_absolute_path):
    """
    Provides Neo4j configuration from graph_configuration.json
    """
    with open(graph_configuration_file_absolute_path, 'r') as f:
        config = json.load(f)
    
    return config