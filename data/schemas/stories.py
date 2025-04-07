from datetime import datetime
from typing import List

from data.schemas.base import BaseSchema

class MomentaryPersonaStory(BaseSchema):
    timestamp: datetime
    story: str


class MomentaryPersonaStories(BaseSchema):
    name_surname: str
    stories: List[MomentaryPersonaStory]


class MomentaryStories(BaseSchema):
    stories: List[MomentaryPersonaStories] 