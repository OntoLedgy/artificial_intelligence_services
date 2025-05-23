import os.path
from pathlib import Path
import pytest

@pytest.fixture(scope='session')
def root_folder_absolute_path():
    base_path = os.path.dirname(
        os.path.abspath(__file__),
    )
    
    return base_path


@pytest.fixture(scope='session')
def test_data_folder_absolute_path(root_folder_absolute_path):
    test_folder_relative_path ="../data"

    test_folder_absolute_path = os.path.normpath(
                os.path.join(
                    root_folder_absolute_path,
                    test_folder_relative_path
                    
                ),
            )
    return test_folder_absolute_path


@pytest.fixture(scope='session')
def outputs_folder_absolute_path(
        test_data_folder_absolute_path):
    outputs_folder_relative_path ="outputs"
    outputs_folder_absolute_path = os.path.join(
            test_data_folder_absolute_path,
            outputs_folder_relative_path
            
            )
    return outputs_folder_absolute_path


@pytest.fixture(scope='session')
def inputs_folder_absolute_path(
    test_data_folder_absolute_path):
    inputs_folder_relative_path ="inputs"
    inputs_folder_absolute_path = os.path.join(
            test_data_folder_absolute_path,
            inputs_folder_relative_path
            
            )
    return inputs_folder_absolute_path

@pytest.fixture(scope='session')
def configurations_folder_absolute_path(root_folder_absolute_path):
    configurations_folder_relative_path = "../configurations"
    
    configurations_folder_absolute_path = os.path.normpath(
            os.path.join(
                    root_folder_absolute_path,
                    configurations_folder_relative_path
                    
                    ),
            )
    return configurations_folder_absolute_path