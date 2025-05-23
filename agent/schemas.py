
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

class UnifiedAgentTurn(BaseModel):
    agent_response: AgentResponse
    execution_result: tuple[dict, str]
    agent_response_2: str

    def __str__(self):
        return f"Thoughts: {self.agent_response.thoughts}\nPython block:\n {self.agent_response.python_block}\nExecution result:\n {self.execution_result}\nAgent response to user:\n {self.agent_response_2}"