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
from agent.utils import load_system_prompt, extract_reply, extract_python_code, format_results
from agent.settings import MAX_TOOL_TURNS, SANDBOX_TIMEOUT

from data.schemas.sft import StaticMemory

# Dataset path
OBSIDIAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_DATASET_PATH = os.path.join(OBSIDIAN_ROOT, "output", "datasets", "base_dataset.json")
BASE_MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "memory", "base_memory")

# Define undesired state exception
class UndesiredStateError(Exception):
    """Exception raised when the state is undesired."""
    pass

def load_dataset() -> Dataset:
    """Load the base dataset from the file."""
    try:
        with open(BASE_DATASET_PATH, "r") as f:
            dataset = json.load(f)
            # Add a "task": "obsidian-retrieval" to each item
            for item in dataset:
                item["task"] = "obsidian-retrieval"
                question = item.get("question")
                if isinstance(question, list) and len(question) == 1:
                    item["question"] = question[0]
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
            
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        super().__init__(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

        # Set the memory path
        self.memory_path = memory_path

    def parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the response from the environment.
        """
        return extract_python_code(response), extract_reply(response)

    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> bool:
        # Get the last message
        last_message = messages[-1]["content"]

        # Get the reply and python code
        python_code, reply = self.parse_response(last_message)

        # Check if the episode should terminate
        python_code_present = len(python_code) > 0
        reply_present = len(reply) > 0

        return (
            reply_present
            and not python_code_present
        )

    def execute_python(self, code: str) -> Tuple[Dict[str, Any], str]:
        locals_dict, error = execute_sandboxed_code(
            code,
            timeout=SANDBOX_TIMEOUT,
            allowed_path=self.memory_path,
            import_module="agent.tools",
        )
        
        return locals_dict, error
    
    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Response from the environment.
        """
        # Get the last message
        last_message = messages[-1]
        print(last_message)
        # last_message is a dict, print keys
        print(last_message.keys())
        last_message = last_message["content"]

        # Get the reply and python code
        python_code, reply = self.parse_response(last_message)
        python_code_present = len(python_code) > 0
        reply_present = len(reply) > 0

        if python_code_present:
            locals_dict, error = self.execute_python(python_code)
            
            return {"role": "user", "content": format_results(locals_dict, error)}
        
        if not python_code_present and reply_present:
            raise UndesiredStateError("The state is undesired. The agent has no reply but is_completed didn't trigger.")
        
        if not python_code_present and not reply_present:
            return {"role": "user", "content": "You need to provide either a <python> block or a <reply> block."}
        