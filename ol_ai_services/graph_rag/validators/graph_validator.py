import networkx as nx
from langchain_core.messages import AIMessage
from langgraph.constants import END

from graph_rag.models.knolwedge_graph_states import KGState

def graph_validator(
        state: KGState) -> KGState:
    print(
            "✅ Graph Validator: Validating knowledge graph")
    G = state["graph"]
    
    validation_report = {
        "num_nodes"   : len(
                G.nodes),
        "num_edges"   : len(
                G.edges),
        "is_connected": nx.is_weakly_connected(
                G) if G.nodes else False,
        "has_cycles"  : not nx.is_directed_acyclic_graph(
                G) if G.nodes else False
        }
    
    state["validation"] = validation_report
    state["messages"].append(
            AIMessage(
                    content=f"Validation report: {validation_report}"))
    print(
            f"   Validation report: {validation_report}")
    
    state["current_agent"] = END
    
    return state

