from typing import List, Dict, Union
import os

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from training.reward.reward import get_reward
from training.reward.folder_dump import dump_folder
from data.schemas.kb import Fact
from agent.utils import log_reward_calculation
from agent.settings import MEMORY_PATH

class MemoryRubric(Rubric):
    def __init__(
            self,
            parser: XMLParser = XMLParser(fields=["thoughts", ("python", "answer")]),
            env_parser: XMLParser = XMLParser(fields=["result"]),
            memory_path: str = MEMORY_PATH,
        ):
        self.parser = parser
        self.env_parser = env_parser
        self.memory_path = memory_path
        self.log_dir = None
        
    def set_memory_path(self, memory_path: str) -> None:
        """
        Set the memory path to use for dumping and reward calculation.
        
        Args:
            memory_path: Path to the memory directory
        """
        self.memory_path = memory_path
    
    def set_log_dir(self, log_dir: str) -> None:
        """
        Set the log directory for reward calculation logs.
        
        Args:
            log_dir: Path to the log directory
        """
        self.log_dir = log_dir
        
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
        if not os.path.exists(self.memory_path):
            # Ensure the directory exists before dumping, or dump_folder might error
            # or return an empty/irrelevant dump for a non-existent path.
            # Depending on dump_folder behavior, might return empty or specific message.
            return "" # Return empty string if memory_dir doesn't exist
        
        return dump_folder(self.memory_path)
        
    def check_facts_reward_func(
            self,
            facts_to_check: List[Dict], # List of dictionaries representing facts
            completion_history=None, # Only used when called directly, not by GRPOTrainer
            # Other batch-level args like prompts, completions are in **kwargs if needed by other funcs
            **kwargs 
    ) -> Union[float, List[Union[float, None]]]: # Returns a single float or a list of rewards for batch
        """
        Reward function that checks if the provided facts are present in the agent's current memory dump.
        
        Args:
            facts_to_check: A list of dictionaries representing facts, or if called in batch mode,
                           a list of lists of dictionaries.
            completion_history: Optional message history, only used when called directly.
            **kwargs: Absorbs other arguments passed by the trainer like 'prompts', 'completions'.
            
        Returns:
            A single reward value or a list of reward values (float or None), one for each sample in a batch.
        """
        # Extract rollout_id from kwargs or completions if available
        rollout_id = None
        if 'prompts' in kwargs and kwargs['prompts'] and len(kwargs['prompts']) > 0:
            # Check if prompt is wrapped with metadata
            prompt_0 = kwargs['prompts'][0]
            if isinstance(prompt_0, dict) and 'rollout_id' in prompt_0:
                rollout_id = prompt_0['rollout_id']
        
        # Also check rollout_ids if provided directly
        if 'rollout_ids' in kwargs and kwargs['rollout_ids'] and len(kwargs['rollout_ids']) > 0:
            rollout_id = kwargs['rollout_ids'][0]
        
        # Determine if this is a batched call from GRPOTrainer or a direct call
        is_batch_call = isinstance(facts_to_check, list) and len(facts_to_check) > 0 and isinstance(facts_to_check[0], list)
        
        if is_batch_call:
            # This is a batched call from GRPOTrainer
            batched_rewards: List[Union[float, None]] = []
            
            for batch_idx, single_sample_facts_as_dicts in enumerate(facts_to_check):
                if not isinstance(single_sample_facts_as_dicts, list):
                    print(f"Warning: Expected List[Dict] for a sample, but got {type(single_sample_facts_as_dicts)}. Assigning 0.0 reward.")
                    batched_rewards.append(0.0)
                    continue

                if not single_sample_facts_as_dicts:  # No fact dictionaries to check for this specific sample
                    batched_rewards.append(0.0)
                    continue
                
                memory_dump_str = self._get_memory_dump_str()
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
                            print(f"Warning: Expected a dict for a fact, but got {type(f_dict)}. Skipping this fact.")
                    
                    if not valid_fact_dicts_found:
                        print(f"Warning: No valid fact dictionaries found in sample. Assigning 0.0 reward.")
                        batched_rewards.append(0.0)
                        continue
                    
                    if not single_sample_facts_as_models:
                        batched_rewards.append(0.0)
                        continue

                    # Calculate reward for this sample
                    reward_for_sample = get_reward(memory_dump_str, single_sample_facts_as_models)
                    batched_rewards.append(float(reward_for_sample))
                    
                    # Log reward calculation if log_dir is set and we have a rollout_id
                    if self.log_dir and rollout_id:
                        log_reward_calculation(
                            self.log_dir,
                            f"{rollout_id}_{batch_idx}",
                            memory_dump_str,
                            single_sample_facts_as_models,
                            float(reward_for_sample)
                        )
                        
                except Exception as e:
                    print(f"Error calculating reward: {e}. Facts data: {single_sample_facts_as_dicts}")
                    batched_rewards.append(0.0)
                    
            return batched_rewards
        else:
            # Direct call from ObsidianAgentEnv.get_rewards()
            if not facts_to_check:
                return 0.0
            
            memory_dump_str = self._get_memory_dump_str()
            if not memory_dump_str:
                return 0.0
            
            try:
                # For direct calls, facts_to_check should already be a list of Fact objects
                # But if it's a list of dicts, convert it to Fact objects
                facts_as_models = []
                for fact in facts_to_check:
                    if isinstance(fact, Fact):
                        facts_as_models.append(fact)
                    elif isinstance(fact, dict):
                        facts_as_models.append(Fact.model_validate(fact))
                    else:
                        print(f"Warning: Invalid fact type: {type(fact)}. Skipping.")
                
                if not facts_as_models:
                    return 0.0
                
                reward = get_reward(memory_dump_str, facts_as_models)
                
                # Log reward calculation
                if self.log_dir and rollout_id:
                    log_reward_calculation(
                        self.log_dir,
                        rollout_id,
                        memory_dump_str,
                        facts_as_models,
                        float(reward)
                    )
                
                return float(reward)
            except Exception as e:
                print(f"Error calculating reward in direct call: {e}")
                return 0.0

# Example of how MemoryRubric might be used within an environment that processes one persona at a time
# (Not directly used by GRPOTrainer in this way, GRPOTrainer calls the reward func directly with batches)
# def evaluate_persona_completion(rubric: MemoryRubric, facts_for_persona: List[Fact]):
#     # In a single-instance evaluation (not batch):
#     # Here, facts_for_persona is List[Fact]
#     # To call the batch-aware version, you'd wrap it:
#     reward_list = rubric.check_facts_reward_func(facts_to_check=[facts_for_persona])
#     return reward_list[0] if reward_list else 0.0
                    