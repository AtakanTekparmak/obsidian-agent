from typing import List, Dict
import json
import os
import shutil
import tempfile

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from training.reward.reward import get_reward
from training.reward.folder_dump import dump_folder
from data.schemas.kb import Fact

class MemoryRubric(Rubric):
    def __init__(
            self,
            parser: XMLParser = XMLParser(fields=["thoughts", ("python", "answer")]),
            env_parser: XMLParser = XMLParser(fields=["result"]),
        ):
        self.parser = parser
        self.env_parser = env_parser
        # self.reward_funcs and self.reward_weights are not strictly needed here
        # if the environment calls the check_facts_reward_func directly.
        # However, if MultiTurnEnv framework relies on these, they might need to be set.
        # For now, let's assume direct call from the custom env.
        
    def get_reward_funcs(self) -> List:
        # This might be required by MultiTurnEnv framework.
        # If so, it should return a list containing a wrapper or the method itself.
        # For now, returning the method directly.
        return [self.check_facts_reward_func] 
    
    def get_reward_weights(self) -> List[float]:
        # Corresponding weight for the reward function.
        return [1.0]
    
    def _get_memory_dump_str(self) -> str:
        """
        Uses dump_folder from training.reward to get the memory dump string.
            
        Returns:
            The memory dump as a string.
        """
        memory_path = "memory_dir" # Should align with agent.settings.MEMORY_PATH
        if not os.path.exists(memory_path):
            # Ensure the directory exists before dumping, or dump_folder might error
            # or return an empty/irrelevant dump for a non-existent path.
            # Depending on dump_folder behavior, might return empty or specific message.
            return "" # Return empty string if memory_dir doesn't exist
        
        return dump_folder(memory_path)
        
    def check_facts_reward_func(
            self,
            completion_history: List[Dict[str, str]], # Represents the chat history for one persona
            facts_to_check: List[Fact],
            **kwargs # Allows for compatibility if MultiTurnEnv passes other args
    ) -> float: # Returns a single reward for the persona's trajectory
        """
        Reward function that checks if the provided facts are present in the agent's memory dump.
        
        Args:
            completion_history: The list of messages in the conversation for the current persona.
                                (Not directly used for dump if it's purely file-based but good for context)
            facts_to_check: A list of Fact objects to check against the memory dump.
            
        Returns:
            A single reward value (0.0 to 1.0) based on how many facts are found.
        """
        if not facts_to_check:
            return 0.0

        memory_dump_str = self._get_memory_dump_str()
        
        if not memory_dump_str:
            return 0.0
        
        try:
            reward = get_reward(memory_dump_str, facts_to_check)
            return float(reward) 
        except Exception as e:
            print(f"Error in check_facts_reward_func: {e}")
            return 0.0
                    