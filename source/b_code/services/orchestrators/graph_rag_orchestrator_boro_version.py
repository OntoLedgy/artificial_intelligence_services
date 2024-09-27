from concurrent.futures import ThreadPoolExecutor, \
    as_completed
from langchain_community.chat_models import ChatOpenAI
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from networkx.algorithms.operators.binary import compose
from networkx.classes import DiGraph
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import (
    run_and_log_function,
    )
from tqdm import tqdm
from typing import List

from configurations.boro_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
    )
from source.b_code.configurations.boro_configurations.nf_open_ai_configurations import (
    NfOpenAiConfigurations,
    )


class BoroGraphRagOrchestrator:
    
    def __init__(
            self,
            data_set,
            model_name = NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI
            ):  # TODO: should we make this a default configuration
        # self.graph = Neo4jGraph()
        self.data_set = data_set
        self.llm = ChatOpenAI(
                api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY,
                temperature=NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE,
                model_name=model_name,
                )
        
        self.llm_transformer = LLMGraphTransformer(
                llm=self.llm,
                # node_properties=["description"],
                # relationship_properties=["description"]
                )
        self.graph_documents = []
    
    
    @run_and_log_function
    def process_text(
            self,
            text: str,
            llm_transformer) -> List[GraphDocument]:
        doc = Document(
            page_content=text)
        
        graph_documents = llm_transformer.convert_to_graph_documents(
                [doc])
        
        return graph_documents
    
    
    # TODO: change parameters to configurations
    @run_and_log_function
    def orchestrate(
            self,
            number_of_rows = NfGeneralConfigurations.NUMBER_OF_ROWS,
            maximum_workers = NfGeneralConfigurations.MAXIMUM_WORKERS,
            ):
        with ThreadPoolExecutor(
                max_workers=maximum_workers) as executor:
            # Submitting all tasks and creating a list of future objects
            if isinstance(
                    self.data_set,
                    list):
                futures = [
                    executor.submit(
                            self.process_text,
                            text_to_be_processed,
                            self.llm_transformer
                            )
                    for text_to_be_processed in self.data_set
                    ]
            
            else:
                futures = [
                    executor.submit(
                            self.process_text,
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
        
        # self.graph.add_graph_documents(
        #     self.graph_documents,
        #     baseEntityLabel=True,
        #     include_source=True
        # )
    
    
    # OXi additions  #####################################
    
    @run_and_log_function
    def get_combined_networkx_graph_from_graph_documents(
            self) -> DiGraph:
        combined_graph = DiGraph()
        
        for i, graph_doc in enumerate(
                self.graph_documents):
            nx_graph = DiGraph()
            
            nodes = getattr(
                graph_doc,
                "nodes",
                [])
            edges = getattr(
                graph_doc,
                "relationships",
                [])
            
            # Add nodes and edges to the NetworkX graph
            for node in nodes:
                # Optionally relabel nodes to avoid collisions
                # nx_graph.add_node(f"{node.id}_graph_{i}")
                nx_graph.add_node(
                    node_for_adding=f"{node.id}",
                    type=node.type)
            
            for edge in edges:
                # Add edges (relabel source and target nodes)
                # source = f"{edge.source.id}_graph_{i}"
                # target = f"{edge.target.id}_graph_{i}"
                source = f"{edge.source.id}"
                target = f"{edge.target.id}"
                nx_graph.add_edge(
                    source,
                    target,
                    type=edge.type)
            
            combined_graph = compose(
                combined_graph,
                nx_graph)
        
        return combined_graph
