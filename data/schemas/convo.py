from typing import List 

from data.schemas.base import BaseSchema
from data.schemas.kb import Fact

class UserChatTurn(BaseSchema):
    turn_number: int
    base_fact: str
    user_message: str
    facts_to_check_so_far: List[Fact]

class MultiTurnConvo(BaseSchema):
    user_persona_name_surname: str
    user_chats: List[UserChatTurn]

class MultiTurnConvos(BaseSchema):
    convos: List[MultiTurnConvo]