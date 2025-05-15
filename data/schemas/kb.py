from datetime import datetime
from typing import List, Optional

from data.schemas.base import BaseSchema
from data.schemas.personas import Persona
from data.schemas.stories import MomentaryPersonaStory


class PersonaWithStories(BaseSchema):
    persona: Persona
    stories: List[MomentaryPersonaStory]


class PersonasWithStories(BaseSchema):
    personas: List[PersonaWithStories]


class Fact(BaseSchema):
    fact_description_or_change: str
    timestamp: Optional[datetime] = None


class PersonalFact(BaseSchema):
    name_surname: str
    facts: List[Fact]


class PersonalFacts(BaseSchema):
    facts: List[PersonalFact]


class KnowledgeBaseItem(BaseSchema):
    persona_with_stories: PersonaWithStories
    facts: List[Fact]


class KnowledgeBase(BaseSchema):
    items: List[KnowledgeBaseItem] 