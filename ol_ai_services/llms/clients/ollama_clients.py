import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable
from ol_ai_services.llms.llm_clients import AbstractLlmClient
from configurations.ol_configurations.nf_ollama_configurations import NfOllamaConfigurations


class OllamaClient(AbstractLlmClient):
    """
    Client for interacting with Ollama API.
    """
    def __init__(
            self,
            model: str = NfOllamaConfigurations.OLLAMA_MODEL,
            temperature: float = NfOllamaConfigurations.OLLAMA_TEMPERATURE,
            base_url: str = NfOllamaConfigurations.OLLAMA_BASE_URL):
        # Normalize model name by adding ":latest" if no tag is specified
        normalized_model = self._normalize_model_name(model)
        
        super().__init__(normalized_model, temperature)
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize Ollama model name by adding ":latest" if no tag is specified.
        
        Args:
            model_name: Original model name
            
        Returns:
            Normalized model name with tag
        """
        if ":" not in model_name:
            return f"{model_name}:latest"
        return model_name

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is already available on the Ollama server.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            True if the model is available, False otherwise
        """
        # Normalize the model name
        normalized_name = self._normalize_model_name(model_name)
        
        # Get available models
        available_models = self.get_available_models()
        
        # Check exact match
        if normalized_name in available_models:
            return True
            
        # Check for model without version tag (some Ollama versions list models differently)
        base_name = normalized_name.split(":")[0]
        for model in available_models:
            if model == base_name or model.startswith(f"{base_name}:"):
                return True
                
        return False
    
    def download_model(self, model_name: str, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        # Normalize the model name
        model_name = self._normalize_model_name(model_name)
        """
        Download a model to the Ollama server if it's not already available.
        
        Args:
            model_name: The name of the model to download
            progress_callback: Optional callback function that receives progress updates
            
        Returns:
            True if the model is successfully downloaded or already available,
            False if there was an error
        """
        # Check if the model is already available
        if self.is_model_available(model_name):
            self.logger.info(f"Model {model_name} is already available")
            return True
            
        self.logger.info(f"Downloading model {model_name}...")
        
        try:
            url = f"{self.base_url}/api/pull"
            
            payload = {"name": model_name}
            
            with requests.post(url, json=payload, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    self.logger.error(f"Failed to download model: {response.status_code}, {response.text}")
                    return False
                    
                # Process the streaming response to track progress
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        progress_data = json.loads(line)
                        
                        # Log progress information
                        if "status" in progress_data:
                            self.logger.info(f"Download status: {progress_data['status']}")
                        
                        # Call the progress callback if provided
                        if progress_callback and callable(progress_callback):
                            progress_callback(progress_data)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing progress data: {e}")
            
            # Verify that the model is now available
            time.sleep(1)  # Brief pause to ensure the model is ready
            return self.is_model_available(model_name)
            
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error while downloading model {model_name}: {str(e)}")
            return False
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout while downloading model {model_name}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error downloading model {model_name}: {str(e)}")
            return False
    
    def get_response(
            self,
            prompt: str,
            max_tokens: int = NfOllamaConfigurations.OLLAMA_MAX_TOKENS,
            auto_download: bool = True) -> str:
        """
        Generate a text response using the Ollama API.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            auto_download: If True, automatically download the model if not available
            
        Returns:
            Generated text response or error message
        """
        try:
            # First check if we can connect to the Ollama server at all
            try:
                health_url = f"{self.base_url}/api/version"
                health_check = requests.get(health_url, timeout=5)
                if health_check.status_code != 200:
                    self.logger.error(f"Ollama server not available: {health_check.status_code}")
                    return f"Error: Ollama server not available at {self.base_url}"
            except requests.exceptions.ConnectionError:
                self.logger.error(f"Cannot connect to Ollama server at {self.base_url}")
                return f"Error: Cannot connect to Ollama server at {self.base_url}"
                
            # Check if the model is available, download if needed and auto_download is True
            if not self.is_model_available(self.model):
                if auto_download:
                    self.logger.info(f"Model {self.model} not found, downloading...")
                    if not self.download_model(self.model):
                        return f"Error: Failed to download model {self.model}"
                else:
                    return f"Error: Model {self.model} is not available and auto_download is disabled"
            
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                    "top_p": NfOllamaConfigurations.OLLAMA_TOP_P,
                    "top_k": NfOllamaConfigurations.OLLAMA_TOP_K,
                    "repeat_penalty": NfOllamaConfigurations.OLLAMA_REPEAT_PENALTY
                }
            }
            
            response = requests.post(
                url, 
                json=payload,
                timeout=NfOllamaConfigurations.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                try:
                    error_msg += f", {response.text}"
                except:
                    pass
                self.logger.error(error_msg)
                return f"Error: {response.status_code}"
                
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to Ollama server: {str(e)}")
            return f"Error: Connection to Ollama server failed"
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout connecting to Ollama server: {str(e)}")
            return f"Error: Timeout connecting to Ollama server"
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from Ollama: {str(e)}")
            return f"Error: Invalid response from Ollama server"
        except Exception as e:
            self.logger.error(f"Error in Ollama client: {str(e)}")
            return f"Error: {str(e)}"
    
    def set_model(self, model: str, auto_download: bool = True):
        """
        Change the model used by the client.
        
        Args:
            model: The model name to use with Ollama
            auto_download: If True, automatically download the model if not available
        """
        # Normalize the model name
        normalized_model = self._normalize_model_name(model)
        
        if normalized_model != self.model:
            if auto_download and not self.is_model_available(normalized_model):
                self.logger.info(f"Model {normalized_model} not found, downloading...")
                self.download_model(normalized_model)
            self.model = normalized_model
    
    def set_temperature(self, temperature):
        """
        Change the temperature parameter for text generation.
        
        Args:
            temperature: Temperature value controlling randomness
        """
        self.temperature = temperature
        
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from the Ollama server.
        
        Returns:
            List of available model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=NfOllamaConfigurations.OLLAMA_TIMEOUT)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                self.logger.error(f"Failed to get Ollama models: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting Ollama models: {str(e)}")
            return []