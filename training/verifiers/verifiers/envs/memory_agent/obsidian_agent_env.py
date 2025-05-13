import inspect
import json
from typing import List, Dict, Any, Callable

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import MEMORY_AGENT_PROMPT
from verifiers.rubrics.memory_rubric import MemoryRubric

from obsidian_agent.engine import execute_sandboxed_code
from obsidian_agent.utils import create_memory_if_not_exists
from obsidian_agent.settings import MEMORY_PATH

# Define constants
STOP_TOKENS = ["</python>", "</answer>"]
MASK_ENV_RESPONSE = True
MAX_STEPS = 10
TOOLS_MODULE = "obsidian_agent.tools"

class ObsidianAgentEnv(MultiTurnEnv):
    """
    Environment for the Obsidian Agent.
    """
    def __init__(
            self,
            dataset: Dataset | None = None,
            eval_dataset: Dataset | None = None,
            tools: List[Callable] = [],
            system_prompt: str = MEMORY_AGENT_PROMPT,
            few_shot: List[Dict[str, str]] = [],
            sampling_args={
                "stop": STOP_TOKENS,
                "include_stop_str_in_output": True
            },
            mask_env_response: bool = MASK_ENV_RESPONSE,
            max_steps: int = MAX_STEPS,
            **kwargs: Any
        ):
        
        # Initialize the environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.llm_parser = XMLParser(fields=["thoughts", ("python", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        self.rubric = MemoryRubric()
        create_memory_if_not_exists()  # Ensure memory path exists

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        """
        Returns a list of reward functions for the environment.
        """
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        """
        Returns a list of reward weights for the environment.
        """
        return self.rubric.get_reward_weights()

    def execute_python_code(self, code: str) -> str:
        """
        Execute the given python code in a sandboxed environment.

        Args:
            code: The python code to execute.

        Returns:
            The output of the python code.
        """
        create_memory_if_not_exists()
        locals_dict, error = execute_sandboxed_code(
            code=code,
            allowed_path=MEMORY_PATH,
            import_module=TOOLS_MODULE
        )
        
        if error:
            return f"Error: {error}"
        else:
            return json.dumps(locals_dict, default=str)
    
    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of steps in the conversation."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count steps from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
                
        return step_count
        
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """
        Determines if the conversation has been completed.
        
        Args:
            messages: The list of messages in the conversation.
            
        Returns:
            True if the conversation has been completed, False otherwise.
        """
        try:
            # Check if we've hit the maximum number of steps
            step_count = self._get_step_count(messages)
            if step_count >= self.max_steps:
                return True
            
            # Check if the last message from the assistant has an answer
            last_assistant_msg = None
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg
                    break
                    
            if last_assistant_msg:
                parsed = self.llm_parser.parse(last_assistant_msg["content"])
                # Consider conversation complete if an answer is provided
                if hasattr(parsed, "answer") and parsed.answer is not None:
                    return True
                
            # If no answer found, conversation is not complete
            return False
            
        except Exception as e:
            # Log and handle any errors
            print(f"Error in is_completed: {e}")
            return False
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generates the environment's response based on the model's action.
        
        Args:
            messages: The list of messages in the conversation.
            
        Returns:
            The environment's response.
        """
        try:
            # Get the last message from the assistant
            last_msg = messages[-1]
            if last_msg["role"] != "assistant":
                return {"role": "user", "content": "Error: Expected assistant message"}
            
            # Parse the message
            parsed = self.llm_parser.parse(last_msg["content"])
            
            # Check if we have Python code to execute
            if hasattr(parsed, "python") and parsed.python is not None:
                # Execute the Python code
                result = self.execute_python_code(parsed.python)
                
                # Format the result
                return {"role": "user", "content": f"<result>\n{result}\n</result>"}
            
            # If the message has an answer but no Python code, it's the agent's final response
            if hasattr(parsed, "answer") and parsed.answer is not None:
                facts_to_check = kwargs.get("facts_to_check_so_far", [])
                if facts_to_check:
                    # Return a message acknowledging the agent's final answer
                    return {"role": "user", "content": "Thanks for your answer. This conversation is now complete."}
                
            # If there's no Python code or answer, prompt the agent to take action
            return {"role": "user", "content": "Please either provide an answer or interact with the memory system."}
                
        except Exception as e:
            # Log and handle any errors
            print(f"Error in env_response: {e}")
            return {"role": "user", "content": f"Error: {str(e)}"}