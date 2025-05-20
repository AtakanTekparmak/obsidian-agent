from typing import List, Dict, Union
import os

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
    
    def _get_memory_dump_str(self, memory_path_to_dump: str) -> str:
        """
        Uses dump_folder from training.reward to get the memory dump string.
            
        Returns:
            The memory dump as a string.
        """
        if not os.path.exists(memory_path_to_dump):
            # Ensure the directory exists before dumping, or dump_folder might error
            # or return an empty/irrelevant dump for a non-existent path.
            # Depending on dump_folder behavior, might return empty or specific message.
            return "" # Return empty string if memory_dir doesn't exist
        
        return dump_folder(memory_path_to_dump)
        
    def check_facts_reward_func(
            self,
            facts_to_check: List[List[Dict]], # Expecting List of Lists of Dictionaries now
            memory_paths: List[str], # Added memory_paths
            # Other batch-level args like prompts, completions are in **kwargs if needed by other funcs
            **kwargs 
    ) -> List[Union[float, None]]: # Must return a list of rewards, one per batch item
        """
        Reward function that checks if the provided facts for each sample in a batch
        are present in the agent's current memory dump.
        
        Args:
            facts_to_check: A list where each element is a List of Dictionaries,
                            each dictionary representing a Fact, for a specific sample in the batch.
            memory_paths: A list of paths to memory directories for each sample in the batch.
            **kwargs: Absorbs other arguments passed by the trainer like 'prompts', 'completions'.
            
        Returns:
            A list of reward values (float or None), one for each sample in the batch.
        """
        batched_rewards: List[Union[float, None]] = []
        
        for i, single_sample_facts_as_dicts in enumerate(facts_to_check): # Iterate over each sample in the batch, get index i
            if not isinstance(single_sample_facts_as_dicts, list):
                print(f"Warning in check_facts_reward_func: Expected List[Dict] for a sample, but got {type(single_sample_facts_as_dicts)}. Assigning 0.0 reward for this sample.")
                batched_rewards.append(0.0)
                continue

            if not single_sample_facts_as_dicts:  # No fact dictionaries to check for this specific sample
                batched_rewards.append(0.0)
                continue
            
            # Get memory dump for this specific sample using its memory_paths entry
            current_sample_memory_path = memory_paths[i]
            memory_dump_str = self._get_memory_dump_str(current_sample_memory_path)

            if not memory_dump_str: # If memory dump is empty, no facts can be found.
                batched_rewards.append(0.0)
                continue
            
            try:
                # Convert list of dicts to list of Fact Pydantic models for this sample
                single_sample_facts_as_models: List[Fact] = []
                valid_fact_dicts_found = False
                for f_dict in single_sample_facts_as_dicts:
                    if isinstance(f_dict, dict):
                        single_sample_facts_as_models.append(Fact.model_validate(f_dict))
                        valid_fact_dicts_found = True
                    else:
                        print(f"Warning in check_facts_reward_func: Expected a dict for a fact, but got {type(f_dict)}. Skipping this particular fact.")
                
                if not valid_fact_dicts_found: # If no valid fact dicts were found to convert
                    # This implies single_sample_facts_as_dicts might have been a list of non-dicts.
                    print(f"Warning in check_facts_reward_func: No valid fact dictionaries found in sample. Assigning 0.0 reward.")
                    batched_rewards.append(0.0)
                    continue
                 
                if not single_sample_facts_as_models: # If conversion resulted in an empty list (e.g. all were invalid non-dicts)
                    # This check is somewhat redundant if valid_fact_dicts_found handles it, but good for safety.
                    batched_rewards.append(0.0)
                    continue

                # 'single_sample_facts_as_models' is now List[Fact] as get_reward expects.
                reward_for_sample = get_reward(memory_dump_str, single_sample_facts_as_models)
                batched_rewards.append(float(reward_for_sample))
            except Exception as e:
                print(f"Error calculating reward for a sample in check_facts_reward_func (after Pydantic conversion attempt): {e}. Facts data: {single_sample_facts_as_dicts}")
                batched_rewards.append(0.0) # Assign 0 reward for this sample due to error
                
        return batched_rewards

# Example of how MemoryRubric might be used within an environment that processes one persona at a time
# (Not directly used by GRPOTrainer in this way, GRPOTrainer calls the reward func directly with batches)
# def evaluate_persona_completion(rubric: MemoryRubric, facts_for_persona: List[Fact]):
#     # In a single-instance evaluation (not batch):
#     # Here, facts_for_persona is List[Fact]
#     # To call the batch-aware version, you'd wrap it:
#     reward_list = rubric.check_facts_reward_func(facts_to_check=[facts_for_persona])
#     return reward_list[0] if reward_list else 0.0
                    