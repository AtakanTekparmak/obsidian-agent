from pydantic import BaseModel

class AgentResponse(BaseModel):
    thoughts: str
    actions_block: str