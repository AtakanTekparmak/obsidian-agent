from typing import List, Dict, Union
import os

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from training.reward.reward import get_reward
from training.reward.folder_dump import dump_folder
from data.schemas.kb import Fact
from agent.settings import get_rollout_memory_path

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
    
    def get_memory_dump_str(self, rollout_id=None) -> str:
        """
        Uses dump_folder from training.reward to get the memory dump string.
        
        Args:
            rollout_id: Optional rollout ID to get memory dump from a specific directory
            
        Returns:
            The memory dump as a string.
        """
        memory_path = get_rollout_memory_path(rollout_id)
        if not os.path.exists(memory_path):
            return ""
        
        return dump_folder(memory_path)
        
    def check_facts_reward_func(
            self,
            facts_to_check: List[Dict],
            rollout_id=None,
            **kwargs 
    ) -> Union[float, None]:
        """
        Reward function that checks if the provided facts are present 
        in the agent's memory dump for a specific rollout.
        
        Args:
            facts_to_check: A list of dictionaries representing Fact objects
            rollout_id: Optional rollout ID to check facts in a specific memory directory
            **kwargs: Absorbs other arguments passed by the trainer
            
        Returns:
            A reward value (float or None)
        """
        # Get memory dump from the specific rollout's directory
        memory_dump_str = self.get_memory_dump_str(rollout_id)
        if not memory_dump_str:
            return 0.0

        # Handle possible different input formats
        if not isinstance(facts_to_check, list):
            return 0.0
        
        # Convert facts to Fact models if they are dictionaries
        facts_as_models = []
        for f in facts_to_check:
            if isinstance(f, dict):
                try:
                    facts_as_models.append(Fact.model_validate(f))
                except Exception as e:
                    print(f"Error converting fact dict to Fact model: {e}")
            elif isinstance(f, Fact):
                facts_as_models.append(f)
        
        if not facts_as_models:
            return 0.0
            
        # Calculate reward using get_reward function
        try:
            reward = get_reward(memory_dump_str, facts_as_models)
            return float(reward)
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0

# Example of how MemoryRubric might be used within an environment that processes one persona at a time
# (Not directly used by GRPOTrainer in this way, GRPOTrainer calls the reward func directly with batches)
# def evaluate_persona_completion(rubric: MemoryRubric, facts_for_persona: List[Fact]):
#     # In a single-instance evaluation (not batch):
#     # Here, facts_for_persona is List[Fact]
#     # To call the batch-aware version, you'd wrap it:
#     reward_list = rubric.check_facts_reward_func(facts_to_check=[facts_for_persona])
#     return reward_list[0] if reward_list else 0.0
                    