from typing import TypedDict, List, Tuple, Dict, Any

class KGState(TypedDict):
    topic: str
    raw_text: str
    entities: List[str]
    relations: List[Tuple[str, str, str]]
    resolved_relations: List[Tuple[str, str, str]]
    graph: Any
    validation: Dict[str, Any]
    messages: List[Any]
    current_agent: str