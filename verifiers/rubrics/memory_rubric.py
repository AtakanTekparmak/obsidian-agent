from typing import List, Dict, Union, Any
import os

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from training.reward.reward import get_reward
from training.reward.folder_dump import dump_folder
from data.schemas.kb import Fact
from agent.settings import get_rollout_memory_path
from agent.utils import log_reward_calculation

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
        
    def _calculate_single_reward(
            self,
            facts_to_check: List[Dict],
            completion_history=None,
            rollout_id=None,
            persona_id=None,
    ) -> float:
        """
        Calculate reward for a single sample.
        
        Args:
            facts_to_check: A list of dictionaries representing Fact objects
            completion_history: The completion history for this sample (optional)
            rollout_id: Optional rollout ID for this sample
            persona_id: Optional persona ID for logging
            
        Returns:
            A reward value as float
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
            reward_value = float(reward)
            
            # Log reward calculation if persona_id is provided
            if persona_id:
                log_reward_calculation(
                    persona_id=persona_id,
                    facts=facts_to_check,
                    memory_dump=memory_dump_str,
                    reward=reward_value,
                    rollout_id=rollout_id
                )
                
            return reward_value
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
        
    def check_facts_reward_func(
            self,
            prompts: List = None,
            completions: List = None,
            facts_to_check: List = None,
            **kwargs
    ) -> List[float]:
        """
        Reward function that processes batched inputs and returns a list of rewards.
        This function follows the interface expected by GRPOEnvTrainer.
        
        Args:
            prompts: List of prompts (optional)
            completions: List of completions (optional)
            facts_to_check: List of facts to check for each sample
            **kwargs: Additional arguments
            
        Returns:
            A list of reward values, one for each completion
        """
        # Debug logging
        completion_count = len(completions) if completions else 0
        facts_count = len(facts_to_check) if facts_to_check else 0
        print(f"check_facts_reward_func called with {completion_count} completions and {facts_count} facts")
        
        # Return early if no facts to check
        if not facts_to_check:
            # Return a list of zeros matching the number of completions
            zeros = [0.0] * len(completions) if completions else []
            print(f"No facts to check, returning {len(zeros)} zeros")
            return zeros
        
        # Process each completion individually to calculate its reward
        results = []
        
        # If completions is provided, calculate a reward for each one
        if completions:
            for i, completion in enumerate(completions):
                # Extract rollout ID from completion metadata
                rollout_id = None
                if isinstance(completion, dict) and "metadata" in completion:
                    metadata = completion.get("metadata", {})
                    if "rollout_id" in metadata:
                        rollout_id = metadata["rollout_id"]
                    elif "sequence_id" in metadata:
                        rollout_id = str(metadata["sequence_id"])
                    elif "batch_id" in metadata:
                        rollout_id = str(metadata["batch_id"])
                    elif "generation_id" in metadata:
                        rollout_id = str(metadata["generation_id"])
                else:
                    # Generate a unique rollout ID if none exists
                    import uuid
                    rollout_id = f"auto_rollout_{str(uuid.uuid4())[:8]}"
                
                # Get persona_id if available for logging
                persona_id = None
                if prompts and i < len(prompts):
                    prompt = prompts[i]
                    if isinstance(prompt, dict) and "persona_id" in prompt:
                        persona_id = prompt["persona_id"]
                
                # Get facts for this completion
                sample_facts = None
                if i < len(facts_to_check):
                    sample_facts = facts_to_check[i]
                    print(f"Processing sample {i} with rollout_id {rollout_id}, facts length: {len(sample_facts) if sample_facts else 0}")
                
                # Calculate the reward for this completion
                try:
                    reward = self._calculate_single_reward(
                        facts_to_check=sample_facts,
                        completion_history=[completion] if isinstance(completion, dict) else None,
                        rollout_id=rollout_id,
                        persona_id=persona_id
                    )
                    results.append(reward)
                except Exception as e:
                    print(f"Error calculating reward for sample {i}: {e}")
                    results.append(0.0)
        else:
            # If no completions provided, return zeros
            results = [0.0] * (len(facts_to_check) if facts_to_check else 0)
            print(f"No completions provided, returning {len(results)} zeros")
        
        print(f"Returning {len(results)} reward values")
        return results

# Example of how MemoryRubric might be used within an environment that processes one persona at a time
# (Not directly used by GRPOTrainer in this way, GRPOTrainer calls the reward func directly with batches)
# def evaluate_persona_completion(rubric: MemoryRubric, facts_for_persona: List[Fact]):
#     # In a single-instance evaluation (not batch):
#     # Here, facts_for_persona is List[Fact]
#     # To call the batch-aware version, you'd wrap it:
#     reward_list = rubric.check_facts_reward_func(facts_to_check=[facts_for_persona])
#     return reward_list[0] if reward_list else 0.0
                    