import json
import os
import tempfile
from typing import Any, Dict, List, Tuple
import uuid

from datasets import Dataset
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser

from training.vf.rubric import MemoryRubric

from agent.engine import execute_sandboxed_code
from agent.utils import load_system_prompt, extract_reply, extract_python_code
from agent.settings import MAX_TOOL_TURNS, SANDBOX_TIMEOUT

from data.schemas.sft import StaticMemory

# Dataset path
OBSIDIAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_DATASET_PATH = os.path.join(OBSIDIAN_ROOT, "output", "datasets", "base_dataset.json")
BASE_MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "memory", "base_memory")

def load_dataset() -> Dataset:
    """Load the base dataset from the file."""
    try:
        with open(BASE_DATASET_PATH, "r") as f:
            dataset = json.load(f)
            # Add a "task": "obsidian-retrieval" to each item
            for item in dataset:
                item["task"] = "obsidian-retrieval"
        return Dataset.from_list(dataset)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {BASE_DATASET_PATH}")

class MemoryEnv(MultiTurnEnv):
    """Environment for the Memory Agent with Python execution."""

    def __init__(
            self,
            system_prompt: str = load_system_prompt(),
            max_turns: int = MAX_TOOL_TURNS,
            memory_path: str = BASE_MEMORY_PATH,
            **kwargs: Any
    ):
        parser = XMLParser(["think", "python", "reply"], answer_field="reply")
        rubric = MemoryRubric(parser=parser)
        
        # Load the dataset
        dataset = load_dataset()
            
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

        # Set the memory path
        self.memory_path = memory_path

    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> bool:
        # Get the last message
        last_message = messages[-1].content

        # Get the reply and python code
        reply = extract_reply(last_message)
        python_code = extract_python_code(last_message)

        # Check if the episode should terminate
        python_code_present = len(python_code) > 0
        reply_present = len(reply) > 0

        return (
            reply_present
            and not python_code_present
        )

    def execute_python(self, code: str) -> str:
        locals_dict, error = execute_sandboxed_code(
            code,
            timeout=SANDBOX_TIMEOUT,
            allowed_path=self.memory_path,
            import_module="agent.tools",
        )
        if error:
            return error
        
        # Combine the locals_dict and error into a dictionary
        result = {
            "locals": locals_dict,
            "error": error
        }
        return str(result)
    
    # TODO
    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Response from the environment.
        """
        pass