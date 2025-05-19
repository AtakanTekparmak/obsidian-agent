import inspect
import json
import os
import shutil # For clearing memory_dir
import uuid # For generating unique rollout IDs
from typing import List, Dict, Any, Callable, Tuple, Optional

from datasets import Dataset, load_dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import MEMORY_AGENT_PROMPT
from verifiers.rubrics.memory_rubric import MemoryRubric
from verifiers.utils.data_utils import preprocess_dataset # For loading convos dataset

from agent.engine import execute_sandboxed_code
from agent.utils import (
    create_memory_if_not_exists, delete_memory, 
    log_reward_calculation, log_completion
)
from agent.settings import get_rollout_memory_path
from data.schemas.kb import Fact # For casting facts_to_check

# Define constants
STOP_TOKENS = ["</python>", "</answer>"] # Keep </answer> for now, might be useful for agent to signal final thought
MASK_ENV_RESPONSE = True # Usually True if env responses are just results/errors
# MAX_STEPS is now implicitly defined by the length of persona conversations
TOOLS_MODULE = "agent.tools" # Assuming tools are in this module for execute_sandboxed_code

class ObsidianAgentEnv(MultiTurnEnv):
    """
    Environment for the Obsidian Agent interacting with conversations from convos.json.
    Each persona's conversation is treated as a trajectory.
    """
    def __init__(
            self,
            convos_dataset_path: str = "training/data/convos.json", # Path to the convos data
            system_prompt: str = MEMORY_AGENT_PROMPT,
            few_shot: List[Dict[str, str]] = [],
            sampling_args={
                "stop": STOP_TOKENS,
                "include_stop_str_in_output": True
            },
            mask_env_response: bool = MASK_ENV_RESPONSE,
            max_steps: Optional[int] = None, # Max steps per persona if needed, else full convo
            **kwargs: Any
        ):
        
        # Load and preprocess the convos dataset
        # The dataset will be a flat list of turns, each with persona_id, facts_to_check, etc.
        full_dataset = preprocess_dataset(name="convos", path=convos_dataset_path)

        super().__init__(
            dataset=full_dataset, # Pass the full flat dataset
            eval_dataset=None, # Or a separate eval slice if desired
            system_prompt=system_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps if max_steps is not None else 0, # If 0 or None, MultiTurnEnv might not limit internally
            sampling_args=sampling_args,
            **kwargs
        )
        
        self.llm_parser = XMLParser(fields=["thoughts", ("python", "answer")])
        self.env_parser = XMLParser(fields=["result"]) # For parsing <result> tags if any
        self.rubric = MemoryRubric()
        
        # Set a unique ID for this environment instance for memory isolation
        self.env_id = str(uuid.uuid4())
        
        # Dictionary mapping rollout IDs to their memory paths
        self.rollout_memories = {}
        
        self.current_persona_id: Optional[str] = None
        self.current_persona_facts_to_check: List[Fact] = []
        
        # Internal state for tracking progress through the flat dataset
        self.current_data_idx = -1 # Index in the flat full_dataset
        self.current_turn_in_persona = 0

    def _get_rollout_id_from_completion(self, completion: Dict[str, str]) -> str:
        """
        Extract a rollout ID from the completion metadata or generate one if not present.
        This is the key method for identifying unique rollouts.
        
        Args:
            completion: The completion dictionary from the model
            
        Returns:
            A unique ID for this rollout
        """
        # If metadata exists and contains a rollout_id, use that
        if isinstance(completion, dict) and completion.get("metadata", {}).get("rollout_id"):
            return completion["metadata"]["rollout_id"]
        
        # If we find a sequence_id, batch_id, or generation_id in metadata, use that
        if isinstance(completion, dict) and "metadata" in completion:
            metadata = completion["metadata"]
            if "sequence_id" in metadata:
                return str(metadata["sequence_id"])
            if "batch_id" in metadata:
                return str(metadata["batch_id"])
            if "generation_id" in metadata:
                return str(metadata["generation_id"])
        
        # Otherwise generate a new ID
        return f"rollout_{str(uuid.uuid4())[:8]}"
    
    def _ensure_rollout_memory(self, rollout_id: str) -> str:
        """
        Ensure a memory directory exists for this rollout and return its path.
        
        Args:
            rollout_id: The rollout ID
            
        Returns:
            The path to the memory directory for this rollout
        """
        if rollout_id not in self.rollout_memories:
            memory_path = create_memory_if_not_exists(rollout_id)
            self.rollout_memories[rollout_id] = memory_path
        return self.rollout_memories[rollout_id]

    def _clear_memory_dir(self, rollout_id: Optional[str] = None):
        """
        Clears the contents of a specific memory directory.
        
        Args:
            rollout_id: The rollout ID to clear, or None to clear the current environment's memory
        """
        if rollout_id is None:
            rollout_id = self.env_id
            
        memory_path = get_rollout_memory_path(rollout_id)
        delete_memory(rollout_id)
        create_memory_if_not_exists(rollout_id)

    def reset_turn(self) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Resets the environment for the next turn or next persona.
        Loads the next user message for the current persona, or starts a new persona.
        This is the primary method for advancing through the dataset and managing persona episodes.
        """
        self.current_data_idx += 1
        
        if self.current_data_idx >= len(self.dataset):
            # End of dataset
            self.done = True
            # Return empty messages and info, or raise an error, or signal completion.
            # MultiTurnEnv's main loop should handle self.done.
            return [{"role": "system", "content": "End of dataset."}], {"status": "dataset_complete"}

        current_turn_data = self.dataset[self.current_data_idx]
        new_persona_id = current_turn_data["persona_id"]

        if self.current_persona_id != new_persona_id:
            # New persona encountered - reset all rollout memories for this persona
            for rollout_id in list(self.rollout_memories.keys()):
                self._clear_memory_dir(rollout_id)
                
            self.current_persona_id = new_persona_id
            self.current_turn_in_persona = 0
            # Load facts for this new persona. They are stored per turn data point but are the same for all turns of a persona.
            # Validate and store them as Fact objects.
            raw_facts = current_turn_data.get("facts_to_check", [])
            self.current_persona_facts_to_check = [Fact.model_validate(f) for f in raw_facts]
        
        self.current_turn_in_persona += 1
        
        # The 'question' from the dataset is the user's message for this turn
        user_message = {"role": "user", "content": current_turn_data["question"]}
        
        # Info to be passed along, could include things like current persona_id for logging
        info = {
            "persona_id": self.current_persona_id,
            "turn_in_persona": self.current_turn_in_persona,
            "is_last_turn_for_persona": current_turn_data["is_last_turn"],
            "current_data_idx": self.current_data_idx
        }
        
        # The format_prompt function in MultiTurnEnv handles adding system prompt and few_shot.
        # We just return the next user message here to be appended.
        return [user_message], info

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def get_rewards(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, float]:
        """
        Calculates rewards at the end of a persona's trajectory.
        This is called when is_completed returns True for the current persona.
        """
        # Get the rollout ID from the last assistant message
        rollout_id = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                rollout_id = self._get_rollout_id_from_completion(msg)
                
                # Log the completion
                if self.current_persona_id:
                    log_completion(
                        persona_id=self.current_persona_id,
                        completion=msg.get("content", ""),
                        rollout_id=rollout_id
                    )
                break
                
        # Ensure the memory path exists for this rollout
        if rollout_id:
            self._ensure_rollout_memory(rollout_id)
            
        # Calculate rewards based on facts in memory
        if self.current_persona_facts_to_check:
            # This uses the specific rollout's memory directory
            reward_val = self.rubric.check_facts_reward_func(
                completion_history=messages,
                facts_to_check=self.current_persona_facts_to_check,
                rollout_id=rollout_id
            )
            
            # Log the reward calculation
            if self.current_persona_id:
                # Get memory dump from the rubric to log it
                memory_dump = self.rubric.get_memory_dump_str(rollout_id)
                log_reward_calculation(
                    persona_id=self.current_persona_id,
                    facts=self.current_persona_facts_to_check,
                    memory_dump=memory_dump,
                    reward=reward_val,
                    rollout_id=rollout_id
                )
            
            return {"memory_fact_check": reward_val}
        return {"memory_fact_check": 0.0} # No facts to check or error

    def execute_python_code(self, code: str, rollout_id: Optional[str] = None) -> str:
        """
        Execute the given python code in a sandboxed environment.
        
        Args:
            code: The Python code to execute
            rollout_id: The rollout ID to use for the memory directory
        """
        # Ensure memory exists for this rollout
        memory_path = self._ensure_rollout_memory(rollout_id if rollout_id else self.env_id)
        
        locals_dict, error = execute_sandboxed_code(
            code=code,
            allowed_path=memory_path,
            import_module=TOOLS_MODULE 
        )
        
        if error:
            # Make error more informative for the agent if possible
            return f"<result>\nError executing code: {str(error)}\n</result>"
        else:
            # Serialize safely
            try:
                result_str = json.dumps(locals_dict, default=str)
            except TypeError:
                result_str = "Error: Result contains non-serializable objects."
            return f"<result>\n{result_str}\n</result>"
        
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """
        Determines if the current persona's conversation has been completed.
        """
        # Check if we have processed a turn from the dataset
        if self.current_data_idx < 0 or self.current_data_idx >= len(self.dataset):
            return True # No data loaded or past end of data

        current_turn_data = self.dataset[self.current_data_idx]
        
        # The trajectory for a persona is complete if 'is_last_turn' is true for the current data point.
        if current_turn_data["is_last_turn"]:
            return True
                
        return False
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generates the environment's response based on the model's last action (typically Python code execution).
        """
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            # This case should ideally be handled by the MultiTurnEnv logic or raise an error.
            return {"role": "user", "content": "<result>\nError: Expected last message to be from assistant.\n</result>"}

        # Extract rollout ID from the completion metadata
        rollout_id = self._get_rollout_id_from_completion(last_msg)
        
        try:
            parsed = self.llm_parser.parse(last_msg["content"])
            
            if hasattr(parsed, "python") and parsed.python is not None:
                # Execute the Python code with the specific rollout's memory directory
                execution_result_str = self.execute_python_code(parsed.python, rollout_id)
                return {"role": "user", "content": execution_result_str}
            
            # If no python and no answer, or just thoughts.
            # Prompt to take action or provide a final answer.
            return {"role": "user", "content": "<result>\nPlease continue. You can use tools or provide a final answer if all tasks for the current user are complete.\n</result>"}
                
        except Exception as e:
            # Log and handle any errors during parsing or execution
            print(f"Error in env_response: {e}")
            return {"role": "user", "content": f"<result>\nError processing your last turn: {str(e)}\n</result>"}

    # Override _get_info if more specific info needs to be returned by step()
    # The base class returns an empty dict.
    def _get_info(self, **kwargs) -> Dict:
        if self.current_data_idx < 0 or self.current_data_idx >= len(self.dataset):
            return {"status": "dataset_complete"}
        
        current_turn_data = self.dataset[self.current_data_idx]
        return {
            "persona_id": self.current_persona_id,
            "turn_in_persona": self.current_turn_in_persona,
            "is_last_turn_for_persona": current_turn_data["is_last_turn"],
            "current_data_idx": self.current_data_idx,
            "facts_for_persona_count": len(self.current_persona_facts_to_check)
        }

    # Override get_termination_reason if needed.
    # Base class returns None or a generic reason if max_steps hit.
    def get_termination_reason(self, messages: List[Dict[str, str]], **kwargs: Any) -> Optional[str]:
        if self.done: # Set in reset_turn when dataset ends
             return "Dataset completed."
        
        if self.current_data_idx >= 0 and self.current_data_idx < len(self.dataset):
            current_turn_data = self.dataset[self.current_data_idx]
            if current_turn_data["is_last_turn"]:
                return f"Completed all turns for persona: {self.current_persona_id}."
            
        return None # No specific termination reason other than what base class might determine