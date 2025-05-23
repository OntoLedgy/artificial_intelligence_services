from langchain_core.messages import AIMessage

from services.graph_rag.models.knolwedge_graph_states import KGState


def data_gatherer(
        state: KGState) -> KGState:
    
    topic = state["topic"]
    
    print(
            f"📚 Data Gatherer: Searching for information about '{topic}'")
    
    if state["raw_text"] == "":
        collected_text = f"{topic} is an important concept. It relates to various entities like EntityA, EntityB, and EntityC. EntityA influences EntityB. EntityC is a type of EntityB."
    else:
        collected_text =(
            state)["raw_text"]
        
    state["messages"].append(
            AIMessage(
                    content=f"Collected raw text about {topic}"))
    
    state["raw_text"] = collected_text
    state["current_agent"] = "entity_extractor"
    
    return state