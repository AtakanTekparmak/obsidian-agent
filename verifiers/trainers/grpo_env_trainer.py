import warnings
from typing import Callable, Optional, Union, Any, List
import uuid

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available
)
from verifiers import RewardFunc
from verifiers.envs.environment import Environment
from verifiers.envs.memory_env import ObsidianAgentEnv
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.imports import LLM, SamplingParams
from verifiers.inference.vllm_client import VLLMClient

# monkey patch vllm client
import trl.extras.vllm_client
trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

if is_wandb_available():
    import wandb



# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        self.vllm_client = None
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.scale_rewards = scale_rewards
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            repetition_penalty=self.repetition_penalty
        )
        self.rollout_envs = {}

    def _create_or_get_env_for_rollout(self, rollout_id: str) -> Environment:
        """
        Create or retrieve an environment instance for a specific rollout.
        For ObsidianAgentEnv, this includes setting a unique memory path.
        
        Args:
            rollout_id: Unique ID for the rollout
            
        Returns:
            Environment instance configured for this rollout
        """
        if rollout_id not in self.rollout_envs:
            # Create a new env instance with the same config as the original env
            if isinstance(self.env, ObsidianAgentEnv):
                # For ObsidianAgentEnv, we need to set a unique memory path
                # Copy all kwargs from original env first
                env_kwargs = {
                    "convos_dataset_path": getattr(self.env, "convos_dataset_path", "training/data/convos.json"),
                    "system_prompt": self.env.system_prompt,
                    "few_shot": self.env.few_shot,
                    "sampling_args": self.env.sampling_args,
                    "mask_env_response": self.env.env_mask == 0,
                    "max_steps": self.env.max_steps,
                }
                
                # Create new env instance
                env = ObsidianAgentEnv(**env_kwargs)
                
                # Set rollout-specific attributes
                env.set_rollout_id(rollout_id)
                
                # Set log_dir in rubric
                if hasattr(env, 'rubric') and hasattr(env, 'log_dir'):
                    env.rubric.set_log_dir(env.log_dir)
                
                self.rollout_envs[rollout_id] = env
            else:
                # For other environment types, just use the original env
                self.rollout_envs[rollout_id] = self.env
                
        return self.rollout_envs[rollout_id]

    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        
        # Add rollout_id to each prompt for tracking
        generation_rollout_ids = []
        prompts_with_rollout_ids = []
        
        if self.accelerator.is_main_process:
            # Generate unique rollout IDs for each batch of generations
            for i in range(0, len(all_prompts), self.num_generations):
                rollout_id = uuid.uuid4().hex[:8]
                generation_rollout_ids.extend([rollout_id] * self.num_generations)
                
                # Create rollout-specific environment
                env = self._create_or_get_env_for_rollout(rollout_id)
                
                # Add rollout metadata to each prompt
                for j in range(i, min(i + self.num_generations, len(all_prompts))):
                    prompt_copy = all_prompts[j].copy()
                    prompt_copy["rollout_id"] = rollout_id
                    prompt_copy["memory_path"] = getattr(env, "memory_path", None)
                    prompts_with_rollout_ids.append(prompt_copy)
                
        if self.accelerator.is_main_process:
            completion_ids = [None] * len(prompts_with_rollout_ids)
            completion_messages = [None] * len(prompts_with_rollout_ids)
            completion_mask = [None] * len(prompts_with_rollout_ids)
            
            # Process each rollout with its dedicated environment
            for i in range(0, len(prompts_with_rollout_ids), self.num_generations):
                rollout_id = prompts_with_rollout_ids[i]["rollout_id"]
                current_batch = prompts_with_rollout_ids[i:i + self.num_generations]
                
                # Get the environment for this rollout
                env = self._create_or_get_env_for_rollout(rollout_id)
                
                # Generate completions using this environment
                env_result = env.generate(
                    prompts=current_batch,
                    llm=self.vllm_client, # type: ignore
                    sampling_params=self.sampling_params,
                )
                
                # Store results for this batch
                for j, (ids, msgs, mask) in enumerate(zip(
                    env_result['ids'], env_result['messages'], env_result['mask']
                )):
                    idx = i + j
                    completion_ids[idx] = ids
                    completion_messages[idx] = msgs
                    completion_mask[idx] = mask
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            generation_rollout_ids = [None] * len(all_prompts)
            prompts_with_rollout_ids = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        generation_rollout_ids = broadcast_object_list(generation_rollout_ids, from_process=0)
        prompts_with_rollout_ids = broadcast_object_list(prompts_with_rollout_ids, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        local_generation_rollout_ids = generation_rollout_ids[process_slice]
        local_prompts_with_rollout_ids = prompts_with_rollout_ids[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)
        
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            
            # Include rollout_id and memory_path in reward function kwargs
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            reward_kwargs["rollout_ids"] = local_generation_rollout_ids
            
            # Instead of using the original prompts, use the ones with rollout IDs
            modified_prompts = local_prompts_with_rollout_ids
            
            output_reward_func = reward_func(prompts=modified_prompts, completions=completions, **reward_kwargs) # type: ignore
            
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()} # type: ignore
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx] # type: ignore
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )


        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards)
        
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        if self.scale_rewards:
            # Scale the rewards to be between 0 and 1
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore  
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # type: ignore

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = gather(rewards).cpu().numpy().tolist()
            if self.accelerator.is_main_process and is_rich_available() and is_wandb_available() and len(prompts_to_log) > 0:
                print_prompt_completions_sample(
                    prompts=prompts_to_log,
                    completions=completions_to_log,
                    rewards=rewards_to_log,
                    num_samples=1,
                    generation_group_size=self.num_generations
                )
                if wandb.run is not None:
                    wandb.log({"sampled_generations": wandb.Table(
                        columns=["prompt", "completion", "reward"],
                        data=[(str(p), str(c), str(r)) for p, c, r in zip(
                            prompts_to_log, completions_to_log, rewards_to_log
                        )]
                    )})

        return {
            "per_token_logprobs": self._get_per_token_logps(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep
            ),
            "old_per_token_logprobs": old_per_token_logps,
            "ref_per_token_logprobs": ref_per_token_logps,
            "advantages": advantages,
            "prompt_attention_mask": prompt_mask,
            "prompt_ids": prompt_ids,
            "prompt_length": prompt_ids.shape[1],
            "attention_mask": attention_mask,
            "completion_ids": completion_ids,
            "action_ids": prompt_completion_ids[:, prompt_ids.shape[1]:],
            "rewards": rewards[process_slice],
        }