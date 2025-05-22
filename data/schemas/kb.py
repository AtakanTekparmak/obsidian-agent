from typing import List

from data.schemas.base import BaseSchema
from data.schemas.personas import Persona

class Fact(BaseSchema):
    fact_description: str


class PersonalFact(BaseSchema):
    name_surname: str
    facts: List[Fact]


class PersonalFacts(BaseSchema):
    facts: List[PersonalFact]


class KnowledgeBaseItem(BaseSchema):
    persona: Persona
    facts: List[Fact]


class KnowledgeBase(BaseSchema):
    items: List[KnowledgeBaseItem] 