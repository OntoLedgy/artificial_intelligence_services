from networkx.classes import DiGraph
from networkx.algorithms.operators.binary import compose
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function


@run_and_log_function()
def get_combined_networkx_graph_from_graph_documents(graph_documents) -> DiGraph:
    combined_graph = DiGraph()

    for i, graph_doc in enumerate(graph_documents):
        nx_graph = DiGraph()

        nodes = getattr(graph_doc, "nodes", [])
        edges = getattr(graph_doc, "relationships", [])

        # Add nodes and edges to the NetworkX graph
        for node in nodes:
            # Optionally relabel nodes to avoid collisions
            # nx_graph.add_node(f"{node.id}_graph_{i}")
            nx_graph.add_node(node_for_adding=f"{node.id}", type=node.type)

        for edge in edges:
            # Add edges (relabel ol_ai_services and target nodes)
            # ol_ai_services = f"{edge.ol_ai_services.id}_graph_{i}"
            # target = f"{edge.target.id}_graph_{i}"
            source = f"{edge.source.id}"
            target = f"{edge.target.id}"
            nx_graph.add_edge(source, target, type=edge.type)

        combined_graph = compose(combined_graph, nx_graph)

    return combined_graph