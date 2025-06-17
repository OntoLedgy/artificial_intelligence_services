from typing import List
from pydantic import BaseModel, Field

# Define the Pydantic model for entity extraction
class Entity(
        BaseModel):
    entity_name: str = Field(
        description="The name of the entity")
    entity_type: str = Field(
        description="The type of the entity (Person, Organization, etc.)")


class Entities(
        BaseModel):
    """List of entities extracted from the query"""
    entities: List[Entity] = Field(
        description="List of entities extracted from the query")