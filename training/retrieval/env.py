from typing import Any
import uuid
import os
import json
from enum import Enum

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from pydantic import BaseModel

from agent.utils import extract_reply, extract_python_code, format_results, delete_memory
from agent.engine import execute_sandboxed_code
from agent.settings import MAX_TOOL_TURNS
from agent.tools import create_memory_if_not_exists

from training.reward import get_reward

from data.schemas.kb import Fact
from data.schemas.sft import StaticMemory

class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: Role
    content: str

class Conversation(BaseModel):
    messages: list[ChatMessage]

DEBUG_MODE = True

class RetrievalEnv(BaseTextEnv):

    def __init__(
        self,
        env_config: dict[str, Any] = {},
        extras: dict[str, Any] = {},
    ):
        super().__init__()
        
        # Store configuration for reset
        self.env_config = env_config
        self.extras = extras

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        self.max_turns = extras["max_turns"] if "max_turns" in extras else MAX_TOOL_TURNS
        
        # Debug mode to preserve memory folders and add logging
        self.debug_mode = DEBUG_MODE
        
        self.memory_path = None  # Will be set in reset()
        self.static_memory_data = None
        self.step_count = 0  # Track number of steps taken
        
        # Store static memory data if available
        if "extra_info" in extras and "static_memory" in extras["extra_info"]:
            self.static_memory_data = extras["extra_info"]["static_memory"]
        
        # Perform initial reset
        self.reset()

        self.messages = []
        self.initial_prompt = None
        
    def reset(self):
        """Reset the environment for a new episode."""
        # Clean up any previous memory
        if self.memory_path and os.path.exists(self.memory_path):
            if not self.debug_mode:
                delete_memory(self.memory_path)
        
        # Create a new memory path with absolute path
        # Use the obsidian-agent root directory to ensure consistency
        obsidian_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        memory_dir = os.path.join(obsidian_root, "memory")
        conversation_dir = os.path.join(obsidian_root, "conversations")
        
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir, exist_ok=True)

        if not os.path.exists(conversation_dir):
            os.makedirs(conversation_dir, exist_ok=True)

        agent_uuid = str(uuid.uuid4())
        
        self.memory_path = os.path.join(memory_dir, f"memory_{agent_uuid}")
        self.conversation_path = os.path.join(conversation_dir, f"conversation_{agent_uuid}.json")
        
        # Reset step counter
        self.step_count = 0

        # Reset messages but preserve initial prompt if it exists
        if hasattr(self, 'initial_prompt') and self.initial_prompt:
            self.messages = []
            # Re-add initial prompt messages
            for message in self.initial_prompt:
                if isinstance(message, dict):
                    role = Role(message["role"])
                    content = message["content"]
                    self.messages.append(ChatMessage(role=role, content=content))
        else:
            self.messages = []
        
        # Create memory directory
        try:
            create_memory_if_not_exists(self.memory_path)
        except Exception as e:
            print(f"Error creating memory at {self.memory_path}: {e}")
            raise
        
        # Load static memory if available
        if self.static_memory_data:
            try:
                static_memory = StaticMemory(**json.loads(self.static_memory_data))
                static_memory.instantiate(self.memory_path)
            except Exception as e:
                print(f"Error instantiating static memory: {e}")
                # Clean up on error
                if os.path.exists(self.memory_path):
                    delete_memory(self.memory_path)
                raise

    def init(self, prompt):
        """Initialize the environment with the initial prompt."""
        # Store the initial prompt
        self.initial_prompt = prompt
        
        # Add initial messages to conversation
        for message in prompt:
            if isinstance(message, dict):
                role = Role(message["role"])
                content = message["content"]
                self.messages.append(ChatMessage(role=role, content=content))
        
        return prompt, {}

    def parse_response(self, action: str) -> tuple[str, str]:
        reply = extract_reply(action)
        python_code = extract_python_code(action)
        return reply, python_code
    
    def is_done(self, action: str) -> bool:
        # Episode is done if agent provides a reply OR max turns reached
        return bool(extract_reply(action)) or self.step_count >= self.max_turns
    
    def save_conversation(self):
        conversation = Conversation(messages=self.messages)
        try:
            with open(self.conversation_path, "w") as f:
                json.dump(conversation.model_dump(mode='json'), f, indent=2)
        except Exception as e:
            print(f"Error saving conversation to {self.conversation_path}: {e}")
            raise
    
    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.messages.append(ChatMessage(role=Role.ASSISTANT, content=action))
        # Increment step counter
        self.step_count += 1
        
        # Parse the response
        reply, python_code = self.parse_response(action)
        python_code_present = len(python_code) > 0
        reply_present = len(reply) > 0

        # Initialize variables for execution results
        local_vars = {}
        error_msg = ""
        env_response = ""

        # Execute the python code if present
        if python_code_present:
            local_vars, error_msg = execute_sandboxed_code(
                code=python_code,
                allowed_path=self.memory_path,
                import_module="agent.tools"
            )
            # Add environment response to messages
            env_response = format_results(local_vars, error_msg)
            self.messages.append(ChatMessage(role=Role.USER, content=env_response))

        # Check if we should terminate
        if (reply_present or self.step_count >= self.max_turns) and not python_code_present:
            # Get the ground truth
            ground_truth = str(self.ground_truth).strip()
            
            # Calculate reward based on whether agent replied and if reply contains ground truth
            if reply_present:
                reward_bool = get_reward(
                    agent_reply=reply,
                    ground_truth=ground_truth,
                    debug=self.debug_mode
                )
                # Convert boolean to float reward (1.0 for correct, 0.0 for incorrect)
                reward = 1.0 if reward_bool else 0.1
            else:
                # No reply after max turns, assign 0 reward
                reward = 0.0

            # Delete the memory unless in debug mode
            if not self.debug_mode:
                delete_memory(self.memory_path)

            self.save_conversation()
            return BaseTextEnvStepOutput(
                observations=[],
                done=True,
                reward=reward,
                metadata={"reply": reply, "max_turns_reached": self.step_count >= self.max_turns}
            )
        else:
            if python_code_present and not reply_present:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": env_response}],
                    done=False,
                    reward=0.1,
                    metadata={"python_code": python_code, "env_response": env_response, "step": self.step_count}
                )
            else:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "Wrong format. Please provide either a </reply> or </python> block."}],
                    done=False,
                    reward=0,
                    metadata={"step": self.step_count}
                )