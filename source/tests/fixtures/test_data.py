import os
from typing_extensions import LiteralString

import pytest
import pandas

@pytest.fixture(scope='session')
def news_data():
    news = pandas.read_csv(
            "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
        )
    
    return news

@pytest.fixture(scope='session')
def pdf_folder_path(
        test_data_configuration,
        inputs_folder_absolute_path)-> LiteralString | str | bytes:
    
    folder_relative_path = test_data_configuration.get_config(
               section_name="graph_rag_paths",
               key='test_pdf_folder_name',
               skip_validation = True
            )
    
    folder_absolute_path = os.path.join(
            inputs_folder_absolute_path,
            folder_relative_path
            )
    
    return folder_absolute_path


@pytest.fixture(scope='session')
def pdf_file_path(test_data_configuration,
                  pdf_folder_path):
    
    file_relative_path = test_data_configuration.get_config(
               section_name="graph_rag_paths",
               key="test_pdf_file_name",
               skip_validation = True
            )
    
    file_absolute_path = os.path.join(
            pdf_folder_path,
            file_relative_path)
    
    return file_absolute_path