from typing import Any
import uuid
import os
import json

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from pydantic import BaseModel

from agent.utils import extract_reply, extract_python_code, format_results, delete_memory
from agent.engine import execute_sandboxed_code
from agent.settings import MAX_TOOL_TURNS
from agent.tools import create_memory_if_not_exists

from training.reward import get_reward

from data.schemas.kb import Fact
from data.schemas.sft import StaticMemory

class RetrievalEnv(BaseTextEnv):

    def __init__(
        self,
        env_config: dict[str, Any] = {},
        extras: dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        self.max_turns = extras["max_turns"] if "max_turns" in extras else MAX_TOOL_TURNS    
        self.memory_path = f"memory/memory_{uuid.uuid4()}"
        
        # Load the static memory
        if "extra_info" in extras and "static_memory" in extras["extra_info"]:
            static_memory_data = extras["extra_info"]["static_memory"]
            static_memory = StaticMemory(**json.loads(static_memory_data))
            static_memory.instantiate(self.memory_path)
        

    def parse_response(self, action: str) -> tuple[str, str]:
        reply = extract_reply(action)
        python_code = extract_python_code(action)
        return reply, python_code
    
    def is_done(self, action: str) -> bool:
        return True if extract_reply(action) else False
    
    def step(self, action: str) -> BaseTextEnvStepOutput:
        # Parse the response
        reply, python_code = self.parse_response(action)

        # Execute the python code
        if python_code:
            local_vars, error_msg = execute_sandboxed_code(
                code=python_code,
                allowed_path=self.memory_path,
                import_module="agent.tools"
            )

        if self.is_done(action):
            # Get the ground truth
            ground_truth = str(self.ground_truth).strip()

            # Delete the memory
            delete_memory(self.memory_path)

            return BaseTextEnvStepOutput(
                observations=[],
                done=True,
                reward=get_reward(
                    folder_dump_str=reply,
                    facts_to_check=[Fact(fact_description=ground_truth)]
                ),
                metadata={"reply": reply}
            )
        else:
            env_response = format_results(local_vars, error_msg)
            return BaseTextEnvStepOutput(
                observations=[{"role": "tool", "content": env_response}],
                done=False,
                reward=0,
                metadata={"python_code": python_code, "env_response": env_response}
            )