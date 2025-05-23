from langchain_core.messages import HumanMessage

from services.graph_rag.orchestrators.langgraph_builder.build_kg_graph import build_kg_graph

def run_knowledge_graph_pipeline(
        topic,
        initial_state):
    print(
        f"🚀 Starting knowledge graph pipeline for: {topic}")
    
    if initial_state is None:
    
        initial_state = {
            "topic"             : topic,
            "raw_text"          : "",
            "entities"          : [],
            "relations"         : [],
            "resolved_relations": [],
            "graph"             : None,
            "validation"        : {},
            "messages"          : [HumanMessage(
                content=f"Build a knowledge graph about {topic}")],
            "current_agent"     : "data_gatherer"
            }
    
    kg_app = build_kg_graph()
    
    final_state = kg_app.invoke(
        initial_state)
    
    print(
        f"✨ Knowledge graph construction complete for: {topic}")
    
    return final_state
