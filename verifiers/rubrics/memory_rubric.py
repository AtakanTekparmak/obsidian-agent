from typing import List, Dict, Union, Optional
import os

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from training.reward.reward import get_reward
from training.reward.folder_dump import dump_folder
from data.schemas.kb import Fact

# Import the logger
from training.utils import trainer_logger

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
        
    def get_reward_funcs(self, instance_id: Optional[int] = None) -> List:
        # Include instance_id parameter to pass it to the reward function
        return [self.check_facts_reward_func] 
    
    def get_reward_weights(self) -> List[float]:
        # Corresponding weight for the reward function.
        return [1.0]
    
    def _get_memory_dump_str(self, memory_dir: Optional[str] = None) -> str:
        """
        Uses dump_folder from training.reward to get the memory dump string.
            
        Args:
            memory_dir: Optional specific memory directory to dump from
            
        Returns:
            The memory dump as a string.
        """
        # If a specific memory directory is provided, use that
        memory_path = memory_dir or "memory_dir"
        
        if not os.path.exists(memory_path):
            # Ensure the directory exists before dumping
            return ""
        
        return dump_folder(memory_path)
        
    def check_facts_reward_func(
            self,
            facts_to_check: List[Dict] = None,
            completion_history: List[Dict] = None,
            memory_dir: Optional[str] = None,
            rollout_id: Optional[int] = None,
            **kwargs 
    ) -> Union[float, List[Union[float, None]]]:
        """
        Reward function that checks if the provided facts are present in the agent's memory dump.
        Can be called directly by the environment for a single persona or by the trainer for batch evaluation.
        
        Args:
            facts_to_check: List of fact dictionaries to check, or a batch list for trainer calls
            completion_history: Optional completion history for the current persona
            memory_dir: Optional specific memory directory to check
            rollout_id: Optional rollout ID for logging
            **kwargs: Additional kwargs passed from the environment or trainer
            
        Returns:
            A float reward value for a single sample, or a list of rewards for a batch
        """
        # Handle direct environment call with explicit memory_dir and facts_to_check
        if memory_dir is not None and facts_to_check is not None and not isinstance(facts_to_check[0], list):
            memory_dump_str = self._get_memory_dump_str(memory_dir)
            
            # Convert fact dictionaries to Fact models
            facts_as_models = []
            for f_dict in facts_to_check:
                if isinstance(f_dict, dict):
                    facts_as_models.append(Fact.model_validate(f_dict))
            
            if not facts_as_models:
                return 0.0
                
            reward_value = get_reward(memory_dump_str, facts_as_models)
            
            # Log the reward call if rollout_id is provided
            if rollout_id is not None:
                trainer_logger.log_reward_call(
                    rollout_id=rollout_id,
                    memory_dump=memory_dump_str,
                    facts_to_check=facts_to_check,
                    reward_value=float(reward_value)
                )
                
            return float(reward_value)
            
        # Handle batch calls from trainer (returns a list of rewards)
        batched_rewards: List[Union[float, None]] = []
        
        # If this is a batch call from the trainer
        if facts_to_check is not None and isinstance(facts_to_check, list):
            # In this case, facts_to_check is a batch list where each item is the facts list for a sample
            # We don't have access to the memory_dir for each separate rollout
            # We'll just return zeros for now, since the real rewards are calculated by the environment
            # directly calling this function with explicit memory_dir
            return [0.0] * len(facts_to_check)
                
        return batched_rewards

# Example of how MemoryRubric might be used within an environment that processes one persona at a time
# (Not directly used by GRPOTrainer in this way, GRPOTrainer calls the reward func directly with batches)
# def evaluate_persona_completion(rubric: MemoryRubric, facts_for_persona: List[Fact]):
#     # In a single-instance evaluation (not batch):
#     # Here, facts_for_persona is List[Fact]
#     # To call the batch-aware version, you'd wrap it:
#     reward_list = rubric.check_facts_reward_func(facts_to_check=[facts_for_persona])
#     return reward_list[0] if reward_list else 0.0
                    