from concurrent.futures import ThreadPoolExecutor, \
    as_completed
from tqdm import tqdm
from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
    )
from configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
    )


class GraphRagOrchestrator:
    
    def __init__(
            self,
            data_set,
            model_name = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O
            ):
        self.graph = Neo4jGraph()
        self.data_set = data_set
        self.llm = ChatOpenAI(
                temperature=NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE,
                model_name=model_name,
                )
        
        self.llm_transformer = LLMGraphTransformer(
                llm=self.llm,
                node_properties=["description"],
                relationship_properties=["description"],
                )
        self.graph_documents = []
    
    
    def _process_text(
            self,
            text: str,
            llm_transformer) -> List[GraphDocument]:
        doc = Document(
            page_content=text)
        
        graph_documents = llm_transformer.convert_to_graph_documents(
                [doc])
        
        return graph_documents
    
    
    def orchestrate(
            self,
            number_of_rows = NfGeneralConfigurations.NUMBER_OF_ROWS,
            maximum_workers = NfGeneralConfigurations.MAXIMUM_WORKERS,
            ):
        with ThreadPoolExecutor(
                max_workers=maximum_workers) as executor:
            # Submitting all tasks and creating a list of future objects
            futures = [
                executor.submit(
                        self._process_text,
                        f"{row['title']} {row['text']}",
                        self.llm_transformer,
                        )
                for i, row in self.data_set.head(
                    number_of_rows).iterrows()
                ]
            
            for future in tqdm(
                    as_completed(
                            futures),
                    total=len(
                            futures),
                    desc="Processing documents"
                    ):
                graph_document = future.result()
                
                self.graph_documents.extend(
                    graph_document)
        
        self.graph.add_graph_documents(
                self.graph_documents,
                baseEntityLabel=True,
                include_source=True
                )
