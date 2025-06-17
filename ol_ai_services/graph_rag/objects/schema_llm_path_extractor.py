class SchemaLLMPathExtractor:
    def __init__(self, llm, possible_entities, possible_relations, kg_validation_schema=None, strict=True):
        self.llm = llm
        self.possible_entities = possible_entities
        self.possible_relations = possible_relations
        self.kg_validation_schema = kg_validation_schema
        self.strict = strict