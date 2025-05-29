from agent.engine import execute_sandboxed_code
from agent.model import get_model_response
from agent.utils import load_system_prompt, create_memory_if_not_exists, extract_python_code, format_results
from agent.settings import MEMORY_PATH, SAVE_CONVERSATION_PATH
from agent.schemas import ChatMessage, Role, AgentResponse, UnifiedAgentTurn

from typing import Union

import json
import os
import uuid

class Agent:
    def __init__(self, max_retries: int = 3):
        self.system_prompt = load_system_prompt()
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]
        self.max_retries = max_retries

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

        # Execute the code from the agent's response
        result = ({}, "")
        if response.python_block:
            create_memory_if_not_exists()
            result = execute_sandboxed_code(
                code=extract_python_code(response.python_block),
                allowed_path=MEMORY_PATH,
                import_module="agent.tools"
            )

        remaining_retries = self.max_retries
        while result[1] and remaining_retries > 0 and response.python_block:
            remaining_retries -= 1
            response = get_model_response(
                messages=self.messages,
                schema=AgentResponse
            )
            result = ({}, "")
            if response.python_block:
                create_memory_if_not_exists()
                result = execute_sandboxed_code(
                    code=extract_python_code(response.python_block),
                    allowed_path=MEMORY_PATH,
                    import_module="agent.tools"
                )

        # Add the agent's response to the conversation history
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response.model_dump_json()))

        # Add the result to the conversation history
        response_2 = None
        self._add_message(ChatMessage(role=Role.USER, content=format_results(result)))
        response_2 = get_model_response(messages=self.messages)
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response_2))

        return response if response_2 is None else UnifiedAgentTurn(
            agent_response=response,
            execution_result=result,
            agent_response_2=response_2,
            error=result[1] if result[1] else None
        )

    def save_conversation(self, log: bool = False):
        """
        Save the conversation messages to a JSON file in
        the output/conversations directory.
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
        try:
            with open(file_path, "w") as f:
                json.dump([message.model_dump() for message in messages], f, indent=4)
        except Exception as e:
            if log:
                print(f"Error saving conversation: {e}")
        if log:
            print(f"Conversation saved to {file_path}")