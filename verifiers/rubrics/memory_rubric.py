from typing import List, Dict, Union
import os
import json
import logging
from datetime import datetime

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from training.reward.reward import get_reward
from training.reward.folder_dump import dump_folder
from data.schemas.kb import Fact
from agent.settings import MEMORY_PATH, REWARD_LOG_DIR

# Set up logging
logger = logging.getLogger('memory_rubric')
logger.setLevel(logging.INFO)
# Remove default handlers to avoid duplicate logging
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Only add file handler, no console handler
log_file = os.path.join(REWARD_LOG_DIR, 'memory_rubric.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class MemoryRubric(Rubric):
    def __init__(
            self,
            parser: XMLParser = XMLParser(fields=["thoughts", ("python", "answer")]),
            env_parser: XMLParser = XMLParser(fields=["result"]),
        ):
        self.parser = parser
        self.env_parser = env_parser
        
    def get_reward_funcs(self, memory_path=None) -> List:
        # Wrap the check_facts_reward_func to include the memory_path
        def wrapped_reward_func(*args, **kwargs):
            if memory_path is not None:
                kwargs['memory_path'] = memory_path
            return self.check_facts_reward_func(*args, **kwargs)
        
        wrapped_reward_func.__name__ = "check_facts_reward_func"
        return [wrapped_reward_func]
    
    def get_reward_weights(self) -> List[float]:
        # Corresponding weight for the reward function.
        return [1.0]
    
    def _get_memory_dump_str(self, memory_path=None) -> str:
        """
        Uses dump_folder from training.reward to get the memory dump string.
            
        Args:
            memory_path: Optional custom memory path. If None, uses the default MEMORY_PATH.
            
        Returns:
            The memory dump as a string.
        """
        memory_path = memory_path or MEMORY_PATH
        if not os.path.exists(memory_path):
            # Ensure the directory exists before dumping
            logger.warning(f"Memory path does not exist: {memory_path}")
            return ""
        
        try:
            return dump_folder(memory_path)
        except Exception as e:
            logger.error(f"Error dumping folder {memory_path}: {e}")
            return f"Error dumping folder: {str(e)}"
        
    def check_facts_reward_func(
            self,
            facts_to_check: List[Dict],
            memory_path=None,
            rollout_id=None,
            persona_id=None,
            **kwargs 
    ) -> List[Union[float, None]]:
        """
        Reward function that checks if the provided facts for each sample in a batch
        are present in the agent's current memory dump.
        
        Args:
            facts_to_check: A list where each element is a List of Dictionaries,
                          each dictionary representing a Fact, for a specific sample in the batch.
            memory_path: Optional custom memory path.
            rollout_id: Optional ID for the rollout for logging.
            persona_id: Optional ID for the persona for logging.
            **kwargs: Absorbs other arguments passed by the trainer like 'prompts', 'completions'.
            
        Returns:
            A list of reward values (float or None), one for each sample in the batch.
        """
        batched_rewards: List[Union[float, None]] = []
        
        # Setup a dedicated log file for this specific rollout/persona if available
        rollout_log_path = None
        if rollout_id is not None:
            rollout_log_dir = os.path.join(REWARD_LOG_DIR, f"rollout_{rollout_id}")
            os.makedirs(rollout_log_dir, exist_ok=True)
            rollout_log_path = os.path.join(rollout_log_dir, f"rewards.log")
            
            # Add a file handler for this specific rollout
            rollout_handler = logging.FileHandler(rollout_log_path)
            rollout_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(rollout_handler)
            
            logger.info(f"Processing rewards for rollout_id={rollout_id}, persona_id={persona_id}, memory_path={memory_path}")
        
        memory_dump_str = self._get_memory_dump_str(memory_path)
        
        # Log memory dump for debugging
        if rollout_id is not None and persona_id is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if isinstance(persona_id, list):
                persona_id_str = "_".join(str(p) for p in persona_id)
            else:
                persona_id_str = str(persona_id)
                
            # Create a directory for this rollout
            rollout_log_dir = os.path.join(REWARD_LOG_DIR, f"rollout_{rollout_id}")
            os.makedirs(rollout_log_dir, exist_ok=True)
            
            # Save memory dump
            memory_dump_path = os.path.join(rollout_log_dir, f"memory_dump_{persona_id_str}_{timestamp}.txt")
            try:
                with open(memory_dump_path, 'w', encoding='utf-8') as f:
                    f.write(memory_dump_str)
                logger.info(f"Saved memory dump to {memory_dump_path}")
            except Exception as e:
                logger.error(f"Failed to write memory dump log: {e}")

        for single_sample_facts_as_dicts in facts_to_check: # Iterate over each sample in the batch
            if not isinstance(single_sample_facts_as_dicts, list):
                logger.warning(f"Expected List[Dict] for a sample, but got {type(single_sample_facts_as_dicts)}. Assigning 0.0 reward for this sample.")
                batched_rewards.append(0.0)
                continue

            if not single_sample_facts_as_dicts:  # No fact dictionaries to check for this specific sample
                batched_rewards.append(0.0)
                continue
            
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
                        logger.warning(f"Expected a dict for a fact, but got {type(f_dict)}. Skipping this particular fact.")
                
                if not valid_fact_dicts_found: # If no valid fact dicts were found to convert
                    # This implies single_sample_facts_as_dicts might have been a list of non-dicts.
                    logger.warning(f"No valid fact dictionaries found in sample. Assigning 0.0 reward.")
                    batched_rewards.append(0.0)
                    continue
                 
                if not single_sample_facts_as_models: # If conversion resulted in an empty list (e.g. all were invalid non-dicts)
                    # This check is somewhat redundant if valid_fact_dicts_found handles it, but good for safety.
                    batched_rewards.append(0.0)
                    continue

                # Log facts being checked for reward calculation
                if rollout_id is not None and persona_id is not None:
                    rollout_log_dir = os.path.join(REWARD_LOG_DIR, f"rollout_{rollout_id}")
                    facts_log_path = os.path.join(rollout_log_dir, f"facts_check_{persona_id_str}_{timestamp}.json")
                    try:
                        with open(facts_log_path, 'w') as f:
                            json.dump([fact.model_dump() for fact in single_sample_facts_as_models], f, indent=2)
                        logger.info(f"Saved facts to check to {facts_log_path}")
                    except Exception as e:
                        logger.error(f"Failed to write facts log: {e}")

                # 'single_sample_facts_as_models' is now List[Fact] as get_reward expects.
                reward_for_sample = get_reward(memory_dump_str, single_sample_facts_as_models)
                
                # Log the reward call result
                if rollout_id is not None and persona_id is not None:
                    rollout_log_dir = os.path.join(REWARD_LOG_DIR, f"rollout_{rollout_id}")
                    reward_log_path = os.path.join(rollout_log_dir, f"reward_{persona_id_str}_{timestamp}.txt")
                    try:
                        with open(reward_log_path, 'w') as f:
                            f.write(f"Reward: {reward_for_sample}\n")
                            f.write(f"Facts count: {len(single_sample_facts_as_models)}\n")
                            f.write(f"Memory path: {memory_path}\n")
                        logger.info(f"Saved reward info to {reward_log_path}")
                    except Exception as e:
                        logger.error(f"Failed to write reward log: {e}")
                
                batched_rewards.append(float(reward_for_sample))
                logger.info(f"Reward calculation for rollout {rollout_id}, persona {persona_id}: {reward_for_sample}")
            except Exception as e:
                logger.error(f"Error calculating reward for a sample in check_facts_reward_func: {e}")
                batched_rewards.append(0.0) # Assign 0 reward for this sample due to error
        
        # Remove the rollout-specific handler if it was added
        if rollout_id is not None:
            # Remove the last handler which should be the rollout handler
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
                
        return batched_rewards