from agent.async_agent.async_engine import execute_sandboxed_code
from agent.async_agent.async_model import get_model_response
from agent.utils import load_system_prompt, create_memory_if_not_exists, extract_python_code, format_results
from agent.settings import MEMORY_PATH, SAVE_CONVERSATION_PATH, MAX_TOOL_TURNS
from agent.schemas import ChatMessage, Role, AgentResponse

from typing import Union
import asyncio
import json
import os
import uuid

class AsyncAgent:
    """Async version of the Agent class supporting concurrent operations."""
    
    def __init__(self, max_tool_turns: int = MAX_TOOL_TURNS, memory_path: str = None):
        self.system_prompt = load_system_prompt()
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]
        self.max_tool_turns = max_tool_turns
        # Set memory_path: use provided path or fall back to default MEMORY_PATH
        self.memory_path = memory_path if memory_path is not None else MEMORY_PATH
        # Ensure memory_path is absolute for consistency
        self.memory_path = os.path.abspath(self.memory_path)

    def _add_message(self, message: Union[ChatMessage, dict]):
        """Add a message to the conversation history."""
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
        else:
            raise ValueError("Invalid message type")

    async def chat(self, message: str) -> AgentResponse:
        """
        Chat with the agent asynchronously.

        Args:
            message: The message to chat with the agent.

        Returns:
            The response from the agent.
        """
        # Add the user message to the conversation history
        self._add_message(ChatMessage(role=Role.USER, content=message))

        # Get the response from the agent
        response = await get_model_response(
            messages=self.messages,
            schema=AgentResponse
        )

        # Execute the code from the agent's response
        result = ({}, "")
        if response.python_block and not response.stop_acting:
            create_memory_if_not_exists(self.memory_path)
            result = await execute_sandboxed_code(
                code=extract_python_code(response.python_block),
                allowed_path=self.memory_path,
                import_module="agent.tools"
            )

        # Add the agent's response to the conversation history
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response.model_dump_json()))

        remaining_tool_turns = self.max_tool_turns
        while remaining_tool_turns > 0 and not response.stop_acting:
            self._add_message(ChatMessage(role=Role.USER, content=format_results(result)))
            response = await get_model_response(
                messages=self.messages,
                schema=AgentResponse
            )
            self._add_message(ChatMessage(role=Role.ASSISTANT, content=response.model_dump_json()))
            if response.python_block and not response.stop_acting:
                create_memory_if_not_exists(self.memory_path)
                result = await execute_sandboxed_code(
                    code=extract_python_code(response.python_block),
                    allowed_path=self.memory_path,
                    import_module="agent.tools"
                )
            remaining_tool_turns -= 1

        return response

    async def save_conversation(self, log: bool = False):
        """
        Save the conversation messages to a JSON file asynchronously.
        """
        if not os.path.exists(SAVE_CONVERSATION_PATH):
            os.makedirs(SAVE_CONVERSATION_PATH, exist_ok=True)

        unique_id = uuid.uuid4()
        file_path = os.path.join(SAVE_CONVERSATION_PATH, f"convo_{unique_id}.json")

        # Convert the execution result messages to tool role
        messages = [
            ChatMessage(role=Role.TOOL, content=message.content) if message.content.startswith("<result>") else ChatMessage(role=message.role, content=message.content)
            for message in self.messages 
        ]
        
        # Use asyncio to write file without blocking
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: json.dump([message.model_dump() for message in messages], open(file_path, "w"), indent=4)
            )
            if log:
                print(f"Conversation saved to {file_path}")
        except Exception as e:
            if log:
                print(f"Error saving conversation: {e}")


# Helper function for concurrent agent operations
async def run_agents_concurrently(agents: list[AsyncAgent], messages: list[str]) -> list[AgentResponse]:
    """
    Run multiple agents concurrently with their respective messages.
    
    Args:
        agents: List of AsyncAgent instances
        messages: List of messages (one per agent)
        
    Returns:
        List of AgentResponse objects
    """
    if len(agents) != len(messages):
        raise ValueError("Number of agents must match number of messages")
    
    tasks = [agent.chat(message) for agent, message in zip(agents, messages)]
    return await asyncio.gather(*tasks) 