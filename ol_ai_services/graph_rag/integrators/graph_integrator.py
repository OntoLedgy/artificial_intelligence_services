import networkx as nx
from langchain_core.messages import AIMessage

from graph_rag.models.knolwedge_graph_states import KGState


def graph_integrator(
        state: KGState) -> KGState:
    print(
            "📊 Graph Integrator: Building the knowledge graph")
    G = nx.DiGraph()
    
    for s, p, o in state["resolved_relations"]:
        if not G.has_node(
                s):
            G.add_node(
                    s)
        if not G.has_node(
                o):
            G.add_node(
                    o)
        G.add_edge(
                s,
                o,
                relation=p)
    
    state["graph"] = G
    state["messages"].append(
            AIMessage(
                    content=f"Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges"))
    
    state["current_agent"] = "graph_validator"
    
    return state
