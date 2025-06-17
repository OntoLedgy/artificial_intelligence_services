import re

from langchain_core.messages import AIMessage
from llama_index.program.openai import OpenAIPydanticProgram

from graph_rag.domain_models.entities import Entities
from graph_rag.models.knolwedge_graph_states import KGState
from prompts.graph_rag.general_graph_rag_prompts import prompt_template_entities


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



# Entity extraction helper function for testing
def extract_entities(llm, query_str):
    try:
        entity_extraction = OpenAIPydanticProgram.from_defaults(
            output_cls=Entities,
            prompt_template_str=prompt_template_entities,
            llm=llm
        )
        extracted_entities = entity_extraction(query_str=query_str)
        return extracted_entities.entities if hasattr(extracted_entities, 'entities') else []
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return []