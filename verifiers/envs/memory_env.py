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
    Each turn within a persona's conversation is treated as a separate trajectory by MultiTurnEnv.
    This class adapts ObsidianAgent to that model.
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
        self.flat_dataset = preprocess_dataset(name="convos", path=convos_dataset_path)
        
        # Create a map for quick lookup of turn data by question
        self.question_to_turn_data_map: Dict[str, Dict[str, Any]] = {}
        for turn_data_item in self.flat_dataset:
            question_content = turn_data_item.get("question")
            if question_content:
                self.question_to_turn_data_map[question_content] = turn_data_item
            else:
                # Handle cases where 'question' might be missing or empty if necessary
                print(f"Warning: Turn data item found without a 'question' field: {turn_data_item}")


        super().__init__(
            dataset=self.flat_dataset, # Pass the full flat dataset
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
        self.last_processed_persona_id: Optional[str] = None
        self._last_turn_data_for_info_cache: Optional[Dict[str, Any]] = None

        # Calculate the number of prefix messages (system prompt + few-shot examples)
        self.num_prefix_messages = 0
        if self.system_prompt:
            self.num_prefix_messages += 1
        if self.few_shot:
            self.num_prefix_messages += len(self.few_shot)


    def _get_turn_data(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Retrieves the turn-specific data from self.flat_dataset based on the initial user question in messages.
        """
        if not messages:
            return None

        # The actual user question is after system prompt and few-shot examples.
        # MultiTurnEnv.format_dataset prepends these.
        # messages[0] is system, messages[1...N] are few-shot, messages[N+1] is user question.
        
        initial_user_message_index = self.num_prefix_messages
        
        if len(messages) > initial_user_message_index and messages[initial_user_message_index]["role"] == "user":
            user_question_content = messages[initial_user_message_index]["content"]
            return self.question_to_turn_data_map.get(user_question_content)
        
        # Fallback: try to find first user message if indexing is off (should not happen with format_dataset)
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                # Ensure this is the one matching an entry in our map (initial question)
                # This fallback is less reliable if user can say the same thing later.
                # The primary mechanism relies on knowing the structure from format_dataset.
                if msg["content"] in self.question_to_turn_data_map:
                     # Check if this user message is indeed the one at the expected position
                    if i == initial_user_message_index:
                        return self.question_to_turn_data_map[msg["content"]]
        
        # If no user message found at the expected position or via simple search
        # self.logger.warning(f"Could not reliably identify user question to fetch turn data from messages: {messages}")
        return None

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

    def get_rewards(
        self, 
        batch_messages: List[List[Dict[str, str]]], # Trainer passes a batch of message histories
        **kwargs: Any
    ) -> List[Dict[str, float]]:
        """
        Calculates rewards for a batch of completed trajectories.
        Each trajectory corresponds to a single turn from the convos.json.
        Rewards are based on facts_to_check for the persona of that turn.
        """
        batch_facts_to_check: List[List[Dict[str, Any]]] = []
        processed_successfully_flags: List[bool] = []

        for messages in batch_messages:
            turn_data = self._get_turn_data(messages)
            if turn_data and "facts_to_check" in turn_data:
                # facts_to_check from data_utils is already List[Dict] (serialized Fact models)
                facts_for_current_turn_persona = turn_data["facts_to_check"]
                batch_facts_to_check.append(facts_for_current_turn_persona)
                processed_successfully_flags.append(True)
            else:
                # Add empty list if no facts or turn_data not found, to maintain batch alignment
                batch_facts_to_check.append([]) 
                processed_successfully_flags.append(False)
                # self.logger.warning(f"Could not find turn_data or facts_to_check for messages: {messages}")

        # MemoryRubric.check_facts_reward_func expects List[List[Dict]] for facts_to_check
        # and List[List[Dict[str,str]]] for completion_history (batch_messages)
        # It returns a List[Union[float, None]]
        # This call assumes completion_history is batch_messages, but rubric might only need it for context
        # or might not use it if facts are directly provided. Let's pass it.
        rubric_rewards_list = self.rubric.check_facts_reward_func(
            completion_history=batch_messages, # Rubric might not use this if facts are primary
            facts_to_check=batch_facts_to_check
        )
        
        output_rewards: List[Dict[str, float]] = []
        for i, reward_val in enumerate(rubric_rewards_list):
            if processed_successfully_flags[i] and reward_val is not None:
                output_rewards.append({"memory_fact_check": float(reward_val)})
            else:
                # Assign 0.0 if data wasn't found or rubric returned None
                output_rewards.append({"memory_fact_check": 0.0})
        
        return output_rewards

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
        Determines if the current trajectory (a single turn from convos.json) is completed.
        This is true if the 'is_last_turn' flag for this turn's data is true.
        """
        turn_data = self._get_turn_data(messages)
        self._last_turn_data_for_info_cache = turn_data # Cache for _get_info

        if turn_data:
            return turn_data.get("is_last_turn", True) # Default to True if key missing
        
        # self.logger.warning(f"is_completed: Could not find turn_data for messages: {messages}. Assuming completed.")
        return True # If turn_data couldn't be found, end the trajectory.
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generates the environment's response based on the model's last action (typically Python code execution).
        Also handles memory clearing when a new persona is encountered.
        """
        # Memory clearing logic:
        # This needs to be done carefully if MultiTurnEnv processes batches with mixed personas concurrently.
        # Assuming for now that either batch size is 1, or personas are not mixed in concurrent execution segments
        # impacting self.last_processed_persona_id.
        current_turn_data = self._get_turn_data(messages)
        self._last_turn_data_for_info_cache = current_turn_data # Cache for _get_info

        if current_turn_data:
            current_persona_id = current_turn_data.get("persona_id")
            if current_persona_id and self.last_processed_persona_id != current_persona_id:
                # self.logger.info(f"New persona '{current_persona_id}' detected (was '{self.last_processed_persona_id}'). Clearing memory.")
                self._clear_memory_dir()
                self.last_processed_persona_id = current_persona_id
        
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            # This case should ideally be handled by the MultiTurnEnv logic or raise an error.
            return {"role": "user", "content": "<result>\nError: Expected last message to be from assistant.\n</result>"}

        try:
            parsed = self.llm_parser.parse(last_msg["content"])
            
            if hasattr(parsed, "python") and parsed.python is not None:
                # Get the code from the "```python" and "```" tags
                if "```python" in parsed.python and "```" in parsed.python:
                    code = parsed.python.split("```python")[1].split("```")[0]
                else:
                    code = parsed.python
                # Execute the Python code
                execution_result_str = self.execute_python_code(code)
                return {"role": "user", "content": execution_result_str}
            
            return {"role": "user", "content": "<result>\nPlease continue. You can use tools or provide a final answer if all tasks for the current user are complete.\n</result>"}
                
        except Exception as e:
            # Log and handle any errors during parsing or execution
            print(f"Error in env_response: {e}")
            return {"role": "user", "content": f"<result>\nError processing your last turn: {str(e)}\n</result>"}