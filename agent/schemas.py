
from enum import Enum
from pydantic import BaseModel

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    role: Role
    content: str

class AgentResponse(BaseModel):
    thoughts: str
    python_block: str