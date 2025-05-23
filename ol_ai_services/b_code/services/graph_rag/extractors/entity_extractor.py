import re

from langchain_core.messages import AIMessage

from services.graph_rag.models.knolwedge_graph_states import KGState


def entity_extractor(
        state: KGState) -> KGState:
    
    print(
            "🔍 Entity Extractor: Identifying entities in the text")
    
    text = state["raw_text"]
    
    entities = re.findall(
            r"Entity[A-Z]",
            text)
    
    entities = [state["topic"]] + entities
    
    state["entities"] = list(
            set(
                    entities))
    
    state["messages"].append(
            AIMessage(
                    content=f"Extracted entities: {state['entities']}"))
    print(
            f"   Found entities: {state['entities']}")
    
    state["current_agent"] = "relation_extractor"
    
    return state

