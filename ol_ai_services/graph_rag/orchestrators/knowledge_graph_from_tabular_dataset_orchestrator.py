from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import (
    run_and_log_function,
)
from configurations.ol_configurations.nf_general_configurations import (
    NfGeneralConfigurations,
)
from graph_rag.extractors.extract_graph_documents_from_dataset import extract_knowledge_graph_from_dataset





@run_and_log_function()
def orchestrate_retrieve_knowledge_graph_from_tabular_data_set(
    data_set,
    maximum_workers=NfGeneralConfigurations.MAXIMUM_WORKERS
):

    graph_documents = extract_knowledge_graph_from_dataset(
        data_set = data_set,
        #maximum_workers=maximum_workers,
            )
    
    
    
    return graph_documents









# # TODO: Maybe this orchestrator class should be dismantled. Orchestrators shouldn't be classes, as they put together
# #  very different processes  - DONE
# class BoroGraphRagOrchestrator:
#     def __init__(
#         self,
#         data_set,
#         model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O
#     ):  # TODO: should we make this a default configuration
#         # self.graph = Neo4jGraph()
#         self.data_set = data_set
#
#         self.llm = ChatOpenAI(
#             #api_key=NfOpenAiConfigurations.OPEN_AI_API_KEY,
#             temperature=NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE,
#             model_name=model_name,
#         )
#
#         self.llm_transformer = LLMGraphTransformer(
#             llm=self.llm,
#             node_properties=["description"],
#             relationship_properties=["description"]
#         )
#         self.graph_documents = []
#
#     @run_and_log_function()
#     def process_text(
#             self,
#             text: str,
#             llm_transformer) -> List[GraphDocument]:
#         doc = Document(
#                 page_content=text)
#
#         graph_documents = llm_transformer.convert_to_graph_documents([doc])
#
#         return graph_documents
#
#     # TODO: change parameters to configurations
#     @run_and_log_function()
#     def orchestrate(
#         self,
#         number_of_rows=NfGeneralConfigurations.NUMBER_OF_ROWS,
#         maximum_workers=NfGeneralConfigurations.MAXIMUM_WORKERS,
#     ):
#         with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
#             # Submitting all tasks and creating a list of future objects
#             if isinstance(self.data_set, list):
#                 futures = [
#                     executor.submit(
#                         self.process_text, text_to_be_processed, self.llm_transformer
#                     )
#                     for text_to_be_processed in self.data_set
#                 ]
#
#
#             else:
#                 futures = [
#                     executor.submit(
#                         self.process_text,
#                         f"{row['title']} {row['text']}",
#                         self.llm_transformer,
#                     )
#                     for i, row in self.data_set.head(number_of_rows).iterrows()
#                 ]
#
#             for future in tqdm(
#                 as_completed(futures), total=len(futures), desc="Processing documents"
#             ):
#                 graph_document = future.result()
#
#                 self.graph_documents.extend(graph_document)
#
#         # self.graph.add_graph_documents(
#         #     self.graph_documents,
#         #     baseEntityLabel=True,
#         #     include_source=True
#         # )
#
#     # OXi additions  #####################################
#
#     @run_and_log_function()
#     def get_combined_networkx_graph_from_graph_documents(self) -> DiGraph:
#         combined_graph = DiGraph()
#
#         for i, graph_doc in enumerate(self.graph_documents):
#             nx_graph = DiGraph()
#
#             nodes = getattr(graph_doc, "nodes", [])
#             edges = getattr(graph_doc, "relationships", [])
#
#             # Add nodes and edges to the NetworkX graph
#             for node in nodes:
#                 # Optionally relabel nodes to avoid collisions
#                 # nx_graph.add_node(f"{node.id}_graph_{i}")
#                 nx_graph.add_node(node_for_adding=f"{node.id}", type=node.type)
#
#             for edge in edges:
#                 # Add edges (relabel ol_ai_services and target nodes)
#                 # ol_ai_services = f"{edge.ol_ai_services.id}_graph_{i}"
#                 # target = f"{edge.target.id}_graph_{i}"
#                 ol_ai_services = f"{edge.ol_ai_services.id}"
#                 target = f"{edge.target.id}"
#                 nx_graph.add_edge(ol_ai_services, target, type=edge.type)
#
#             combined_graph = compose(combined_graph, nx_graph)
#
#         return combined_graph
