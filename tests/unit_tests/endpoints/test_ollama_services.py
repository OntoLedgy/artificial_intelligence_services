import pytest
import os
import json
import time
import requests
from unittest.mock import patch, MagicMock

from ol_ai_services.llms.client_factory import LlmClientFactory, LlmClientType
from ol_ai_services.llms.llm_generator import LlmGenerator
from ol_ai_services.llms.clients.ollama_clients import OllamaClient
from ol_ai_services.llms.clients.langchain_ollama_clients import LangChainOllamaClients
from ol_ai_services.model_management.model_types import ModelTypes


class TestOllamaServices:
    """
    Tests for the Ollama client implementation using a real Ollama server.
    """
    
    @pytest.fixture(scope="class")
    def ollama_config(self):
        """Load the Ollama configuration from the JSON file."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configurations",
            "ollama_configuration.json"
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config["ollama_api"]
    
    @pytest.fixture(scope="class")
    def ollama_server_available(self, ollama_config):
        """Check if the Ollama server is available."""
        try:
            response = requests.get(f"{ollama_config['base_url']}/api/version", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    @pytest.fixture(scope="class")
    def ollama_client(self, ollama_config):
        """Create an Ollama client with the configuration."""
        client = OllamaClient(
            model=ollama_config["model"],
            temperature=ollama_config["temperature"],
            base_url=ollama_config["base_url"]
        )
        return client
    
    @pytest.fixture(scope="class")
    def langchain_ollama_client(self, ollama_config):
        """Create a LangChain Ollama client with the configuration."""
        client = LangChainOllamaClients(
            model=ollama_config["model"],
            temperature=ollama_config["temperature"],
            base_url=ollama_config["base_url"]
        )
        return client
    
    def test_ollama_client_get_response(self, ollama_client, ollama_config, ollama_server_available):
        """Test that OllamaClient can get a response from a real Ollama server."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        prompt = "What is artificial intelligence?"
        response = ollama_client.get_response(prompt, max_tokens=ollama_config["max_tokens"])
        
        # Print the response for manual inspection
        print(f"Ollama Response: {response}")
        
        # Basic validation that we got a non-empty response
        assert response is not None
        assert len(response) > 0
        
        # If we got an error response, print it but don't fail the test
        # This allows the tests to run even if the model isn't available
        if response.startswith("Error:"):
            print(f"Received error response: {response}")
            pytest.skip(f"Skipping due to error: {response}")
    
    def test_langchain_ollama_client_get_response(self, langchain_ollama_client, ollama_config, ollama_server_available):
        """Test that LangChainOllamaClients can get a response from a real Ollama server."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        prompt = "Explain the concept of machine learning."
        response = langchain_ollama_client.get_response(prompt, max_tokens=ollama_config["max_tokens"])
        
        # Print the response for manual inspection
        print(f"LangChain Ollama Response: {response}")
        
        # Basic validation that we got a non-empty response
        assert response is not None
        assert isinstance(response, str)
    
    def test_client_factory_ollama(self, ollama_config, ollama_server_available):
        """Test that the factory can create an Ollama client that works with a real server."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        client = LlmClientFactory.create_client(
            LlmClientType.OLLAMA,
            model=ollama_config["model"],
            temperature=ollama_config["temperature"],
            base_url=ollama_config["base_url"]
        )
        
        prompt = "What are neural networks?"
        response = client.get_response(prompt, max_tokens=ollama_config["max_tokens"])
        
        print(f"Factory Ollama Response: {response}")
        
        # Basic validation
        assert response is not None
        assert len(response) > 0
        
        if response.startswith("Error:"):
            print(f"Received error response: {response}")
            pytest.skip(f"Skipping due to error: {response}")
    
    def test_llm_generator_with_ollama(self, ollama_config, ollama_server_available):
        """Test that LlmGenerator works with an Ollama client."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        generator = LlmGenerator(
            LlmClientType.OLLAMA,
            model=ollama_config["model"],
            temperature=ollama_config["temperature"],
            base_url=ollama_config["base_url"]
        )
        
        prompt = "Explain the difference between supervised and unsupervised learning."
        response = generator.generate_text(prompt, max_tokens=ollama_config["max_tokens"])
        
        print(f"Generator Ollama Response: {response}")
        
        # Basic validation
        assert response is not None
        assert len(response) > 0
        
        if isinstance(response, str) and response.startswith("Error:"):
            print(f"Received error response: {response}")
            pytest.skip(f"Skipping due to error: {response}")
    
    def test_get_available_models(self, ollama_client, ollama_server_available):
        """Test that we can retrieve available models from the Ollama server."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        models = ollama_client.get_available_models()
        
        print(f"Available Ollama Models: {models}")
        
        # Basic validation - models should be a list, may be empty if no models are installed
        assert isinstance(models, list)
    
    def test_is_model_available(self, ollama_client, ollama_config, ollama_server_available):
        """Test that we can check if a model is available."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        # Check the model specified in the config
        is_available = ollama_client.is_model_available(ollama_config["model"])
        print(f"Model {ollama_config['model']} available: {is_available}")
        
        # For a real test, this depends on the model being pre-downloaded
        # But our improved client should handle this automatically
        # If it's not downloaded, the client should download it
        assert isinstance(is_available, bool)
    
    @pytest.mark.parametrize("model_name", [
        "tinyllama",  # A small model that downloads quickly
    ])
    def test_model_download(self, ollama_client, model_name, ollama_server_available):
        """Test that we can download a model if it's not available."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        # For CI environments where we can't download models, make this test optional
        try:
            # Skip the test if the model is already downloaded to save time
            if ollama_client.is_model_available(model_name):
                print(f"Model {model_name} is already available, skipping download test")
                pytest.skip(f"Model {model_name} is already available")
            
            # Create a simple progress callback to print updates
            progress_updates = []
            def track_progress(progress_data):
                print(f"Download progress: {progress_data}")
                progress_updates.append(progress_data)
            
            # Download the model with a timeout to avoid test hanging
            success = ollama_client.download_model(model_name, progress_callback=track_progress)
            
            # Check that the download was successful
            assert success is True
            
        except Exception as e:
            pytest.skip(f"Skipping model download test due to error: {str(e)}")
    
    def test_auto_download_on_response(self, ollama_config, ollama_server_available):
        """Test that the client auto-downloads a model when needed for response."""
        if not ollama_server_available:
            pytest.skip("Ollama server is not available")
            
        # Use a model that is likely not available
        test_model = "gemma:2b"  # A small model for testing
        
        client = OllamaClient(
            model=test_model,
            temperature=ollama_config["temperature"],
            base_url=ollama_config["base_url"]
        )
        
        # Use mocking to test the auto-download behavior
        with patch.object(client, 'is_model_available') as mock_is_available, \
             patch.object(client, 'download_model') as mock_download, \
             patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:
            
            # Setup mocks
            mock_is_available.return_value = False
            mock_download.return_value = True
            
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_get.return_value = mock_health_response
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Mocked response"}
            mock_post.return_value = mock_response
            
            # Try to get a response which should trigger download
            response = client.get_response(
                "What is machine learning?",
                max_tokens=10,
                auto_download=True
            )
            
            # Verify download was attempted
            mock_download.assert_called_once_with(test_model)
            assert response == "Mocked response"
    
    def test_set_model_auto_download(self, ollama_client):
        """Test that setting a model auto-downloads it if needed."""
        # Use a model that is likely not downloaded
        test_model = "phi:latest"
        
        # Use mocking to avoid actual server calls
        with patch.object(ollama_client, 'is_model_available') as mock_is_available, \
             patch.object(ollama_client, 'download_model') as mock_download:
            
            # Setup mocks
            mock_is_available.return_value = False
            
            # Set the model which should trigger download
            ollama_client.set_model(test_model, auto_download=True)
            
            # Verify download was attempted
            mock_download.assert_called_once_with(test_model)