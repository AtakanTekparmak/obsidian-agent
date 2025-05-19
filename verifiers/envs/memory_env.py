import inspect
import json
import os
import shutil # For clearing memory_dir
import logging
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
from agent.settings import MEMORY_PATH, get_rollout_memory_path, LOG_DIR, REWARD_LOG_DIR
from data.schemas.kb import Fact # For casting facts_to_check

# Define constants
STOP_TOKENS = ["</python>", "</answer>"] # Keep </answer> for now, might be useful for agent to signal final thought
MASK_ENV_RESPONSE = True # Usually True if env responses are just results/errors
# MAX_STEPS is now implicitly defined by the length of persona conversations
TOOLS_MODULE = "agent.tools" # Assuming tools are in this module for execute_sandboxed_code

# Set up logging directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REWARD_LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger('obsidian_agent_env')
logger.setLevel(logging.INFO)
# Remove default handlers to avoid duplicate logging
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Add file handler
log_file = os.path.join(LOG_DIR, 'obsidian_agent_env.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

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
            rollout_id: Optional[str] = None, # ID for the rollout to create a unique memory directory
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
        
        # Store the rollout ID for logging
        self.rollout_id = rollout_id
        # Get a unique memory path for this rollout
        self.memory_path = get_rollout_memory_path(rollout_id)
        
        # Setup rollout-specific logging if needed
        if rollout_id:
            rollout_log_dir = os.path.join(LOG_DIR, f"rollout_{rollout_id}")
            os.makedirs(rollout_log_dir, exist_ok=True)
            rollout_log_file = os.path.join(rollout_log_dir, "env.log")
            
            # Add a file handler for this specific rollout
            rollout_handler = logging.FileHandler(rollout_log_file)
            rollout_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(rollout_handler)
            
        logger.info(f"Initialized ObsidianAgentEnv with rollout_id={rollout_id}, memory_path={self.memory_path}")
        
        # Create the memory directory
        self._create_memory_dir()
        
        self.current_persona_id: Optional[str] = None
        self.current_persona_facts_to_check: List[Fact] = []
        
        # Internal state for tracking progress through the flat dataset
        self.current_data_idx = -1 # Index in the flat full_dataset
        self.current_turn_in_persona = 0

    def _create_memory_dir(self):
        """Creates the memory directory for this rollout."""
        if not os.path.exists(self.memory_path):
            os.makedirs(self.memory_path, exist_ok=True)
            logger.info(f"Created memory directory: {self.memory_path}")

    def _clear_memory_dir(self):
        """Clears the contents of the memory directory for this rollout."""
        if os.path.exists(self.memory_path):
            for item in os.listdir(self.memory_path):
                item_path = os.path.join(self.memory_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            logger.info(f"Cleared memory directory: {self.memory_path}")
        self._create_memory_dir()

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
            logger.info(f"Started new persona {new_persona_id} with {len(self.current_persona_facts_to_check)} facts to check")
        
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
        return self.rubric.get_reward_funcs(memory_path=self.memory_path)
    
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
            
            # Log the facts being checked and memory state for debugging
            if self.rollout_id:
                rollout_log_dir = os.path.join(REWARD_LOG_DIR, f"rollout_{self.rollout_id}")
                os.makedirs(rollout_log_dir, exist_ok=True)
                facts_log_path = os.path.join(rollout_log_dir, f"facts_{self.current_persona_id}.json")
                try:
                    with open(facts_log_path, 'w') as f:
                        json.dump([fact.model_dump() for fact in self.current_persona_facts_to_check], f, indent=2)
                    logger.info(f"Saved persona facts to {facts_log_path}")
                except Exception as e:
                    logger.error(f"Failed to write facts file: {e}")
            
            reward_val = self.rubric.check_facts_reward_func(
                completion_history=messages, # Pass the history
                facts_to_check=self.current_persona_facts_to_check,
                memory_path=self.memory_path,
                rollout_id=self.rollout_id,
                persona_id=self.current_persona_id
            )
            logger.info(f"Reward for persona {self.current_persona_id} (rollout {self.rollout_id}): {reward_val}")
            return {"memory_fact_check": reward_val}
        return {"memory_fact_check": 0.0} # No facts to check or error

    def execute_python_code(self, code: str) -> str:
        """
        Execute the given python code in a sandboxed environment.
        """
        # Ensure memory exists (though it should from __init__ and _clear_memory_dir)
        self._create_memory_dir()
        
        locals_dict, error = execute_sandboxed_code(
            code=code,
            allowed_path=self.memory_path,
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

        try:
            parsed = self.llm_parser.parse(last_msg["content"])
            
            if hasattr(parsed, "python") and parsed.python is not None:
                # Execute the Python code
                execution_result_str = self.execute_python_code(parsed.python)
                return {"role": "user", "content": execution_result_str}
            
            # If no python and no answer, or just thoughts.
            # Prompt to take action or provide a final answer.
            return {"role": "user", "content": "<result>\nPlease continue. You can use tools or provide a final answer if all tasks for the current user are complete.\n</result>"}
                
        except Exception as e:
            # Log and handle any errors during parsing or execution
            logger.error(f"Error in env_response: {e}")
            return {"role": "user", "content": f"<result>\nError processing your last turn: {str(e)}\n</result>"}

    def _get_info(self, **kwargs) -> Dict:
        if self.current_data_idx < 0 or self.current_data_idx >= len(self.dataset):
            return {"status": "dataset_complete"}
        
        current_turn_data = self.dataset[self.current_data_idx]
        return {
            "persona_id": self.current_persona_id,
            "turn_in_persona": self.current_turn_in_persona,
            "is_last_turn_for_persona": current_turn_data["is_last_turn"],
            "current_data_idx": self.current_data_idx,
            "facts_for_persona_count": len(self.current_persona_facts_to_check),
            "rollout_id": self.rollout_id,
            "memory_path": self.memory_path
        }

    def get_termination_reason(self, messages: List[Dict[str, str]], **kwargs: Any) -> Optional[str]:
        if self.done: # Set in reset_turn when dataset ends
             return "Dataset completed."
        
        if self.current_data_idx >= 0 and self.current_data_idx < len(self.dataset):
            current_turn_data = self.dataset[self.current_data_idx]
            if current_turn_data["is_last_turn"]:
                return f"Completed all turns for persona: {self.current_persona_id}."
            
        return None # No specific termination reason other than what base class might determine