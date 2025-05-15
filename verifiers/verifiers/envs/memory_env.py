import inspect
import json
import os
import shutil # For clearing memory_dir
from typing import List, Dict, Any, Callable, Tuple, Optional

from datasets import Dataset, load_dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import MEMORY_AGENT_PROMPT
from verifiers.rubrics.memory_rubric import MemoryRubric
from verifiers.utils.data_utils import preprocess_dataset # For loading convos dataset

from agent.engine import execute_sandboxed_code
from agent.utils import create_memory_if_not_exists
from agent.settings import MEMORY_PATH
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
        
        create_memory_if_not_exists()
        self.current_persona_id: Optional[str] = None
        self.current_persona_facts_to_check: List[Fact] = []
        
        # Internal state for tracking progress through the flat dataset
        self.current_data_idx = -1 # Index in the flat full_dataset
        self.current_turn_in_persona = 0

    def _clear_memory_dir(self):
        """Clears the contents of the MEMORY_PATH directory."""
        if os.path.exists(MEMORY_PATH):
            for item in os.listdir(MEMORY_PATH):
                item_path = os.path.join(MEMORY_PATH, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        create_memory_if_not_exists() # Recreate if it was removed

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
            # New persona encountered
            self._clear_memory_dir()
            self.current_persona_id = new_persona_id
            self.current_turn_in_persona = 0
            # Load facts for this new persona. They are stored per turn data point but are the same for all turns of a persona.
            # Validate and store them as Fact objects.
            raw_facts = current_turn_data.get("facts_to_check", [])
            self.current_persona_facts_to_check = [Fact.model_validate(f) for f in raw_facts]
        
        self.current_turn_in_persona += 1
        
        # The 'question' from the dataset is the user's message for this turn
        user_message = {"role": "user", "content": current_turn_data["question"]}
        
        # We need to construct the message history to send to the agent.
        # If it's the first turn of a persona, it's just system + user_message.
        # Otherwise, it's the existing self.messages + user_message.
        # MultiTurnEnv's `self.messages` should handle history accumulation.
        # This method's responsibility is to provide the *next* input.
        # The base MultiTurnEnv's step() will append this user_message to self.messages.

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
        # This check ensures rewards are calculated only when a persona's dialogue is complete.
        # It relies on `is_completed` correctly identifying the end of a persona's session.
        if self.current_persona_facts_to_check:
            # The `check_facts_reward_func` from MemoryRubric expects the full completion history
            # for the persona and the facts to check for *that specific persona*.
            # `messages` here is the accumulated conversation for the current persona.
            reward_val = self.rubric.check_facts_reward_func(
                completion_history=messages, # Pass the history
                facts_to_check=self.current_persona_facts_to_check
            )
            return {"memory_fact_check": reward_val}
        return {"memory_fact_check": 0.0} # No facts to check or error

    def execute_python_code(self, code: str) -> str:
        """
        Execute the given python code in a sandboxed environment.
        """
        # Ensure memory exists (though it should from __init__ and _clear_memory_dir)
        create_memory_if_not_exists() 
        
        locals_dict, error = execute_sandboxed_code(
            code=code,
            allowed_path=MEMORY_PATH,
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
            
        # Optional: Add max_steps check *within* a persona if self.max_steps is set meaningfully
        if self.max_steps and self.max_steps > 0: # max_steps from MultiTurnEnv constructor
             # We need a way to count assistant turns *for the current persona*
             # This is tricky as `messages` is the full history up to this point for the persona.
             # A simpler approach is to rely on `is_last_turn` from data.
             # If `self.max_steps` (from `MultiTurnEnv`) is used, it might prematurely end a persona.
             # For now, let's prioritize `is_last_turn`.
             pass

        # If the agent provides a final answer (e.g., via <answer> tag and no python)
        # this could also signify completion, but `is_last_turn` is more deterministic here.
        # last_assistant_msg = next((msg for msg in reversed(messages) if msg["role"] == "assistant"), None)
        # if last_assistant_msg:
        #     parsed = self.llm_parser.parse(last_assistant_msg["content"])
        #     if hasattr(parsed, "answer") and parsed.answer is not None and \
        #        (not hasattr(parsed, "python") or parsed.python is None):
        #         # If this logic is enabled, ensure it aligns with `is_last_turn`
        #         # or decide which takes precedence.
        #         # return True 
        #         pass
                
        return False
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generates the environment's response based on the model's last action (typically Python code execution).
        """
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            # This case should ideally be handled by the MultiTurnEnv logic or raise an error.
            return {"role": "user", "content": "<result>\nError: Expected last message to be from assistant.\n</result>"}

        try:
            parsed = self.llm_parser.parse(last_msg["content"])
            
            if hasattr(parsed, "python") and parsed.python is not None:
                # Execute the Python code
                execution_result_str = self.execute_python_code(parsed.python)
                return {"role": "user", "content": execution_result_str}
            
            # If agent gives an <answer> without python, and it's NOT the last turn of persona,
            # we might want to prompt it to continue or use tools if appropriate.
            # However, if `is_completed` handles the end-of-persona, this becomes simpler.
            # If it's the last turn, `is_completed` will return True, and `get_rewards` is called.
            # The flow doesn't necessarily need a special env_response for a non-tool answer.
            # MultiTurnEnv will just proceed to the next turn (which would be a new persona or end).
            
            # If no python and no answer, or just thoughts.
            # Prompt to take action or provide an answer.
            # This can be a generic prompt if the agent is expected to always use python or provide a final answer.
            # If the agent is just thinking, this response will be its next input.
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
        
        # If MultiTurnEnv's max_steps is used and hit:
        # step_count = self._get_step_count(messages) # Assuming _get_step_count counts assistant turns
        # if self.max_steps and self.max_steps > 0 and step_count >= self.max_steps:
        #    return f"Reached max_steps ({self.max_steps}) for persona."
            
        return None # No specific termination reason other than what base class might determine