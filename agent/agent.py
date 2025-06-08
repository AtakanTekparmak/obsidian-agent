from agent.engine import execute_sandboxed_code
from agent.model import get_model_response, create_openai_client, create_instructor_client
from agent.utils import load_system_prompt, create_memory_if_not_exists, extract_python_code, format_results
from agent.settings import MEMORY_PATH, SAVE_CONVERSATION_PATH, MAX_TOOL_TURNS
from agent.schemas import ChatMessage, Role, AgentResponse

from typing import Union

import json
import os
import uuid

class Agent:
    def __init__(self, max_tool_turns: int = MAX_TOOL_TURNS, memory_path: str = None):
        self.system_prompt = load_system_prompt()
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]
        self.max_tool_turns = max_tool_turns
        
        # Each Agent instance gets its own clients to avoid bottlenecks
        self._client = create_openai_client()
        self._instructor_client = create_instructor_client(self._client)
        
        # Set memory_path: use provided path or fall back to default MEMORY_PATH
        if memory_path is not None:
            # Always place custom memory paths inside a "memory/" folder
            self.memory_path = os.path.join("memory", memory_path)
        else:
            # Use default MEMORY_PATH but also place it inside "memory/" folder
            self.memory_path = os.path.join("memory", MEMORY_PATH)
            
        # Ensure memory_path is absolute for consistency
        self.memory_path = os.path.abspath(self.memory_path)

    def _add_message(self, message: Union[ChatMessage, dict]):
        """ Add a message to the conversation history. """
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
        else:
            raise ValueError("Invalid message type")

    def chat(self, message: str) -> AgentResponse:
        """
        Chat with the agent.

        Args:
            message: The message to chat with the agent.

        Returns:
            The response from the agent.
        """
        # Add the user message to the conversation history
        self._add_message(ChatMessage(role=Role.USER, content=message))

        # Get the response from the agent using this instance's clients
        response = get_model_response(
            messages=self.messages,
            schema=AgentResponse,
            client=self._client,
            instructor_client=self._instructor_client
        )

        # Execute the code from the agent's response
        result = ({}, "")
        if response.python_block and not response.stop_acting:
            create_memory_if_not_exists(self.memory_path)
            result = execute_sandboxed_code(
                code=extract_python_code(response.python_block),
                allowed_path=self.memory_path,
                import_module="agent.tools"
            )

        # Add the agent's response to the conversation history
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response.model_dump_json()))

        remaining_tool_turns = self.max_tool_turns
        while remaining_tool_turns > 0 and not response.stop_acting:
            self._add_message(ChatMessage(role=Role.USER, content=format_results(result)))
            response = get_model_response(
                messages=self.messages,
                schema=AgentResponse,
                client=self._client,
                instructor_client=self._instructor_client
            )
            self._add_message(ChatMessage(role=Role.ASSISTANT, content=response.model_dump_json()))
            if response.python_block and not response.stop_acting:
                create_memory_if_not_exists(self.memory_path)
                result = execute_sandboxed_code(
                    code=extract_python_code(response.python_block),
                    allowed_path=self.memory_path,
                    import_module="agent.tools"
                )
            remaining_tool_turns -= 1

        return response

    def save_conversation(
            self, 
            log: bool = False,
            save_folder: str = None
        ):
        """
        Save the conversation messages to a JSON file in
        the output/conversations directory.
        """
        if not os.path.exists(SAVE_CONVERSATION_PATH):
            os.makedirs(SAVE_CONVERSATION_PATH, exist_ok=True)

        unique_id = uuid.uuid4()
        if save_folder is None:
            file_path = os.path.join(SAVE_CONVERSATION_PATH, f"convo_{unique_id}.json")
        else:
            folder_path = os.path.join(SAVE_CONVERSATION_PATH, save_folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, f"convo_{unique_id}.json")

        # Convert the execution result messages to tool role
        messages = [
            ChatMessage(role=Role.TOOL, content=message.content) if message.content.startswith("<r>") else ChatMessage(role=message.role, content=message.content)
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