from langgraph.graph import StateGraph, END

from services.graph_rag.extractors.entity_extractor import entity_extractor
from services.graph_rag.extractors.relation_extractor import relation_extractor
from services.graph_rag.integrators.graph_integrator import graph_integrator
from services.graph_rag.models.knolwedge_graph_states import KGState
from services.graph_rag.resolvers.entity_resolver import entity_resolver
from services.graph_rag.validators.graph_validator import graph_validator
from services.text_extraction.data_gatherer import data_gatherer

def router(
        state: KGState) -> str:
    return state["current_agent"]


def build_kg_graph():
    workflow = StateGraph(
            KGState)
    
    workflow.add_node(
            "data_gatherer",
            data_gatherer)
    
    workflow.add_node(
            "entity_extractor",
            entity_extractor)
    
    workflow.add_node(
            "relation_extractor",
            relation_extractor)
    
    workflow.add_node(
            "entity_resolver",
            entity_resolver)
    
    workflow.add_node(
            "graph_integrator",
            graph_integrator)
    
    workflow.add_node(
            "graph_validator",
            graph_validator)
    
    workflow.add_conditional_edges(
            "data_gatherer",
            router,
            {
                "entity_extractor": "entity_extractor"
                })
    
    workflow.add_conditional_edges(
            "entity_extractor",
            router,
            {
                "relation_extractor": "relation_extractor"
                })
    
    workflow.add_conditional_edges(
            "relation_extractor",
            router,
            {
                "entity_resolver": "entity_resolver"
                })
    
    workflow.add_conditional_edges(
            "entity_resolver",
            router,
            {
                "graph_integrator": "graph_integrator"
                })
    
    workflow.add_conditional_edges(
            "graph_integrator",
            router,
            {
                "graph_validator": "graph_validator"
                })
    
    workflow.add_conditional_edges(
            "graph_validator",
            router,
            {
                END: END
                })
    
    workflow.set_entry_point(
            "data_gatherer")
    
    return workflow.compile()