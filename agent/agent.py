from agent.engine import execute_sandboxed_code
from agent.model import get_model_response
from agent.utils import load_system_prompt, create_memory_if_not_exists, extract_python_code, format_results
from agent.settings import MEMORY_PATH
from agent.schemas import ChatMessage, Role, AgentResponse, UnifiedAgentTurn

from typing import Union

class Agent:
    def __init__(self):
        self.system_prompt = load_system_prompt()
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]

    def _add_message(self, message: Union[ChatMessage, dict]):
        """ Add a message to the conversation history. """
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
        else:
            raise ValueError("Invalid message type")

    def chat(self, message: str) -> Union[AgentResponse, UnifiedAgentTurn]:
        """
        Chat with the agent.

        Args:
            message: The message to chat with the agent.

        Returns:
            The response from the agent.
        """
        # Add the user message to the conversation history
        self._add_message(ChatMessage(role=Role.USER, content=message))

        # Get the response from the agent
        response = get_model_response(
            messages=self.messages,
            schema=AgentResponse
        )
        # Add the agent's response to the conversation history
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response.model_dump_json()))

        # Execute the code from the agent's response
        result = ({}, "No code to execute, no result to return.")
        if response.python_block:
            create_memory_if_not_exists()
            result = execute_sandboxed_code(
                code=extract_python_code(response.python_block),
                allowed_path=MEMORY_PATH,
                import_module="agent.tools"
            )

        # Add the result to the conversation history
        response_2 = None
        self._add_message(ChatMessage(role=Role.USER, content=format_results(result)))
        response_2 = get_model_response(messages=self.messages)
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response_2))

        return response if response_2 is None else UnifiedAgentTurn(
            agent_response=response,
            execution_result=result,
            agent_response_2=response_2
        )