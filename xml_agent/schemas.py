
from enum import Enum
from pydantic import BaseModel
from typing import Optional

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ChatMessage(BaseModel):
    role: Role
    content: str

class AgentResponse(BaseModel):
    thoughts: str
    python_block: Optional[str] = None
    reply: Optional[str] = None

    def __str__(self):
        return f"Thoughts: {self.thoughts}\nPython block:\n {self.python_block}\nReply: {self.reply}"