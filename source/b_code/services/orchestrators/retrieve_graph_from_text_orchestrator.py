import os.path
from langchain_experimental.graph_transformers import LLMGraphTransformer
from networkx.classes import DiGraph
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from configurations.constants import PDF_FILE_EXTENSION
from services.data_preparation.pdf_services import extract_text_from_pdf
from services.llms.chat_open_ai_session_getter import get_chat_open_ai_session
from services.orchestrators.graph_rag_orchestrator_boro_version import BoroGraphRagOrchestrator


def orchestrate_retrieve_graph_from_text(
        text_path: str) \
        -> DiGraph:
    # TODO: maybe use the Texts class created in the huggingface restructuring branch?
    text = \
        __get_text_from_file(
            text_path=text_path)
    
    llm_graph_transformer = \
        __get_llm_graph_transformer()
    
    graph_rag_orchestrator = \
        BoroGraphRagOrchestrator(
            model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI,
            data_set=text)
    
    graph_documents = \
        graph_rag_orchestrator.process_text(
            text=text,
            llm_transformer=graph_rag_orchestrator.llm_transformer)
    
    graph_rag_orchestrator.graph_documents = \
        graph_documents
    
    networkx_graph = \
        graph_rag_orchestrator.get_combined_networkx_graph_from_graph_documents()
    
    return \
        networkx_graph


def __get_text_from_file(
        text_path: str) \
        -> str:
    text_file_path_without_extension, text_file_extension = \
        os.path.splitext(
                text_path)
    
    # TODO: Only PDF texts implemented so far. DOCX needs to be added, and maybe more.
    if text_file_extension == PDF_FILE_EXTENSION:
        text = \
            extract_text_from_pdf(
                    pdf_path=text_path)
    
    else:
        text = \
            str()
    
    return \
        text


def __get_llm_graph_transformer() \
        -> LLMGraphTransformer:
    chat_open_ai_session = \
        get_chat_open_ai_session(
                api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY,
                temperature=NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE,
                model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI)
    
    llm_graph_transformer = \
        LLMGraphTransformer(
                llm=chat_open_ai_session)
    
    return \
        llm_graph_transformer
