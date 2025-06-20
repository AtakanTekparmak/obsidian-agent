import json
import os
import tempfile
from typing import Any, Dict, List, Tuple, Union

from datasets import Dataset
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser

from training.retrieval.memory_rubric import MemoryRubric
from agent.engine import execute_sandboxed_code
from agent.utils import load_system_prompt
from agent.settings import MAX_TOOL_TURNS, SANDBOX_TIMEOUT

class MemoryEnv(MultiTurnEnv):
    """Environment for the Memory Agent with Python execution."""

    def __init__(
            self,
            dataset: Dataset | List[Dict] | None = None,
            system_prompt: str = load_system_prompt(),
            max_turns: int = MAX_TOOL_TURNS,
            **kwargs: Any
    ):
        parser = XMLParser(["think", "python", "reply"], answer_field="reply")
        self.env_parser = XMLParser(["result"])
        rubric = MemoryRubric(parser=parser, env_parser=self.env_parser)
        
        # Handle dataset parameter - convert to Dataset if needed
        if dataset is not None:
            processed_dataset = Dataset.from_list(dataset) if not isinstance(dataset, Dataset) else dataset
        else:
            processed_dataset = None
            
        super().__init__(
            dataset=processed_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> bool:
        return self.parser.parse_answer(messages) is not None

    def _setup_memory(self, state: Dict[str, Any]) -> str:
        """Create a temporary memory directory and populate static files."""
        tmpdir = tempfile.TemporaryDirectory()
        memory_dir = tmpdir.name
        state["_tmpdir"] = tmpdir  # keep reference to prevent GC
        state["memory_dir"] = memory_dir
        if "static_memory" in state:
            try:
                data = json.loads(state["static_memory"])
                guideline = data.get("guideline", "")
                if guideline:
                    with open(os.path.join(memory_dir, "guideline.md"), "w") as f:
                        f.write(guideline)
                user_path = data.get("user_file_path")
                user_content = data.get("user_file_content", "")
                if user_path:
                    full = os.path.join(memory_dir, user_path)
                    os.makedirs(os.path.dirname(full), exist_ok=True)
                    with open(full, "w") as f:
                        f.write(user_content)
            except Exception:
                pass
        return memory_dir

    def execute_python(self, code: str, memory_dir: str) -> str:
        locals_dict, error = execute_sandboxed_code(
            code,
            timeout=SANDBOX_TIMEOUT,
            allowed_path=memory_dir,
            import_module="agent.tools",
        )
        if error:
            return error
        return str(locals_dict)

    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        if "memory_dir" not in state:
            memory_dir = self._setup_memory(state)
        else:
            memory_dir = state["memory_dir"]
        parsed = self.parser.parse(messages[-1]["content"])
        if hasattr(parsed, "python") and parsed.python:
            result = self.execute_python(parsed.python, memory_dir)
            return {"role": "user", "content": self.env_parser.format(result=result)}, state
        return {"role": "user", "content": ""}, state