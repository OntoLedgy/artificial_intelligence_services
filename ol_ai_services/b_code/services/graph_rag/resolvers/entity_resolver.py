from langchain_core.messages import AIMessage

from services.graph_rag.models.knolwedge_graph_states import KGState


def entity_resolver(
        state: KGState) -> KGState:
    print(
            "🔄 Entity Resolver: Resolving duplicate entities")
    
    entity_map = {}
    for entity in state["entities"]:
        canonical_name = entity.lower().replace(
                " ",
                "_")
        entity_map[entity] = canonical_name
    
    resolved_relations = []
    for s, p, o in state["relations"]:
        s_resolved = entity_map.get(
                s,
                s)
        o_resolved = entity_map.get(
                o,
                o)
        resolved_relations.append(
                (s_resolved, p, o_resolved))
    
    state["resolved_relations"] = resolved_relations
    state["messages"].append(
            AIMessage(
                    content=f"Resolved relations: {resolved_relations}"))
    
    state["current_agent"] = "graph_integrator"
    
    return state

