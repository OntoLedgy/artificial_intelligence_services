import re

from langchain_core.messages import AIMessage

from graph_rag.models.knolwedge_graph_states import KGState


def relation_extractor(
        state: KGState) -> KGState:
    print(
            "🔗 Relation Extractor: Identifying relationships between entities")
    text = state["raw_text"]
    entities = state["entities"]
    relations = []
    
    relation_patterns = [
        (r"([A-Za-z]+) relates to ([A-Za-z]+)", "relates_to"),
        (r"([A-Za-z]+) influences ([A-Za-z]+)", "influences"),
        (r"([A-Za-z]+) is a type of ([A-Za-z]+)", "is_type_of")
        ]
    
    for e1 in entities:
        for e2 in entities:
            if e1 != e2:
                for pattern, rel_type in relation_patterns:
                    if re.search(
                            f"{e1}.*{rel_type}.*{e2}",
                            text.replace(
                                    "_",
                                    " "),
                            re.IGNORECASE) or \
                            re.search(
                                    f"{e1}.*{e2}",
                                    text,
                                    re.IGNORECASE):
                        relations.append(
                                (e1, rel_type, e2))
    
    state["relations"] = relations
    state["messages"].append(
            AIMessage(
                    content=f"Extracted relations: {relations}"))
    print(
            f"   Found relations: {relations}")
    
    state["current_agent"] = "entity_resolver"
    
    return state

