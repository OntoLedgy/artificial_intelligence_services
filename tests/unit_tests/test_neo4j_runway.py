import os

import pytest

from graph_rag.orchestrators.neo4j_runway.utils.data import load_local_files
from ol_ai_services.graph_rag.orchestrators.neo4j_runway import Discovery, GraphDataModeler, PyIngest, UserInput
from ol_ai_services.graph_rag.orchestrators.neo4j_runway.code_generation import PyIngestConfigGenerator
from ol_ai_services.graph_rag.orchestrators.neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM


class TestLangchainGraphRetriever:
    @pytest.fixture(autouse=True)
    def setup_method(self,
                     inputs_folder_absolute_path):
        self.input_directory = os.path.join(
                inputs_folder_absolute_path,
                "countries/"
                )

    def test_load_countries_data(self):
    
        data_directory = self.input_directory
      
        data_dictionary = {
                        'id': 'unique id for a country.',
                        'name': 'the country name.',
                        'phone_code': 'country area code.',
                        'capital': 'the capital of the country.',
                        'currency_name': "name of the country's currency.",
                        'region': 'primary region of the country.',
                        'subregion': 'subregion location of the country.',
                        'timezones': 'timezones contained within the country borders.',
                        'latitude': 'the latitude coordinate of the country center.',
                        'longitude': 'the longitude coordinate of the country center.'
                        }
        
        use_cases = [
                "Which region contains the most subregions?",
                "What currencies are most popular?",
                "Which countries share timezones?"
            ]
        
        data = load_local_files(data_directory=data_directory,
                                data_dictionary=data_dictionary,
                                general_description="This is data on countries and their attributes.",
                                use_cases=use_cases,
                                include_files=["countries.csv"])
        
        llm_disc = OpenAIDiscoveryLLM(
                model_name='gpt-4o-mini-2024-07-18',
                model_params={"temperature": 0})
        
        llm_dm = OpenAIDataModelingLLM(
                model_name='gpt-4o-2024-05-13',
                model_params={"temperature": 0.5})
        
        disc = Discovery(
                llm=llm_disc,
                data=data)
        
        disc.run(
                show_result=True,
                notebook=True)
        
        gdm = GraphDataModeler(
            llm=llm_dm,
            discovery=disc)
        
        gdm.create_initial_model()
        
        gdm.current_model.visualize()
        
        gdm.iterate_model(
            corrections="Create a Capital node from the capital property.")
        
        gdm.current_model.visualize()
        
        gen = PyIngestConfigGenerator(
            data_model=gdm.current_model,
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD"),
            uri="bolt://localhost:7687",
            database="neo4jrunway",
            file_directory=data_directory,
            source_name="countries.csv")
        
        pyingest_yaml = gen.generate_config_string()
        
        PyIngest(
            config=pyingest_yaml,
            verbose=False)
        
        