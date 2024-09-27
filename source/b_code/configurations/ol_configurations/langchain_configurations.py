from pydantic import BaseModel


class LangChainConfiguration(
        BaseModel):
    langchain_url: str
    langchain_api_key: str
    langchain_model_id: str
    langchain_model_version: str
    langchain_input_format: str
    langchain_output_format: str
