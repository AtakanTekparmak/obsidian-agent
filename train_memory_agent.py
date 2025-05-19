import os
import json
import uuid
from pathlib import Path
from datasets import Dataset
import argparse

from verifiers.envs.memory_env import ObsidianAgentEnv
from verifiers.trainers import GRPOEnvTrainer
from verifiers.utils.data_utils import preprocess_dataset, format_dataset
from trl import GRPOConfig
from agent.settings import LOG_DIR, REWARD_LOG_DIR


# GPU setup for 8 GPUs (e.g., 2 for vLLM inference, 6 for training)
# This is an example, adjust gpu allocation and model paths as needed.
"""
8 GPU setup
# Terminal 1: Start vLLM server 

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 16384 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

# Terminal 2: Launch training script 

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml train_memory_agent.py 
"""


# Define constants
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "results/memory_agent_qwen_2.5_7b"

def main():
    parser = argparse.ArgumentParser(description="Train Memory Agent with GRPO and ObsidianAgentEnv")
    parser.add_argument("--data_path", type=str, default="training/data/convos.json", 
                        help="Path to conversation data JSON file (convos.json)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, 
                        help="Directory to save model and results")
    parser.add_argument("--lr", type=float, default=5e-6, 
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Per device batch size for training")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of epochs for training")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, # Replace with a local HF model for vLLM
                        help="Model to use for training (must be a local model path for vLLM)")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction of data to use for testing/evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting")
    parser.add_argument("--num_generations", type=int, default=4, # Number of trajectories per prompt
                        help="Number of generations per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REWARD_LOG_DIR, exist_ok=True)
    
    data_path = Path(args.data_path).resolve()
    print(f"Loading and preprocessing data from {data_path} using 'convos' preset...")
    
    # Load the full dataset using the "convos" preset from preprocess_dataset
    # This will return a flat Dataset with 'question', 'persona_id', 'is_last_turn', 'facts_to_check', etc.
    full_dataset = preprocess_dataset(name="convos", path=str(data_path))
    
    # Split the dataset into training and evaluation sets
    if args.test_size > 0:
        dataset_splits = full_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
        train_ds = dataset_splits["train"]
        eval_ds = dataset_splits["test"]
        print(f"Train dataset size: {len(train_ds)}, Eval dataset size: {len(eval_ds)}")
    else:
        train_ds = full_dataset
        eval_ds = None # Or a small subset if evaluation is still desired
        print(f"Train dataset size: {len(train_ds)}, No evaluation set.")

    # Generate a unique base rollout ID for this training run
    base_rollout_id = str(uuid.uuid4())[:8]
    print(f"Using base rollout ID: {base_rollout_id}")
    
    # Initialize the environment with a rollout ID
    # Each parallel environment will get a unique rollout ID based on this one
    env = ObsidianAgentEnv(
        convos_dataset_path=str(data_path),
        # Pass the base rollout ID - the env will use this to create a unique memory directory
        rollout_id=f"{base_rollout_id}_main"
    )

    # Format datasets to include the 'prompt' field expected by the trainer
    print("Formatting datasets...")
    train_ds = format_dataset(
        train_ds,
        system_prompt=env.system_prompt,
        question_key="question", # 'question' is the key from preprocess_convos_persona
        answer_key="answer"      # 'answer' is also present
    )
    if eval_ds:
        eval_ds = format_dataset(
            eval_ds,
            system_prompt=env.system_prompt,
            question_key="question",
            answer_key="answer"
        )

    # Initialize the GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=f"memory_agent_{Path(args.model_name).name}_{args.lr}",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        num_train_epochs=args.epochs,
        use_vllm=True, 
        temperature=0.7, # Default, adjust as needed
        top_p=1.0,       # Default, adjust as needed
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=20 if eval_ds else 0, # Example: evaluate every 100 steps
        save_strategy="steps",
        save_steps=20, # Example: save every 200 steps
        logging_steps=10,
        report_to="wandb", # if configured
        # vLLM server details if use_vllm=True
        vllm_server_host="0.0.0.0",
        vllm_server_port=8000,
        bf16=True,
        reward_weights=env.get_reward_weights()
    )
    
    # Custom callback to create unique environments for each rollout
    class EnvCreationCallback:
        def __init__(self, base_rollout_id, data_path):
            self.base_rollout_id = base_rollout_id
            self.data_path = data_path
            
        def create_env(self, rollout_index):
            """Create a new environment instance with a unique rollout ID"""
            rollout_id = f"{self.base_rollout_id}_{rollout_index}"
            return ObsidianAgentEnv(
                convos_dataset_path=self.data_path,
                rollout_id=rollout_id
            )
    
    env_creator = EnvCreationCallback(base_rollout_id, str(data_path))
    
    # Initialize the trainer with a custom env_factory
    # This will ensure each rollout gets a unique environment with its own memory directory
    trainer = GRPOEnvTrainer(
        model=args.model_name, 
        env=env,  # This is the main env used for configuration, actual envs will be created per rollout
        reward_funcs=env.get_reward_funcs(),
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    
    # Override the env generation in trainer
    original_env_generate = trainer.env.generate
    
    def wrapped_env_generate(prompts, llm, sampling_params, **kwargs):
        """Wrap the env.generate method to create unique environments per batch"""
        # Get the current process index to make environments unique across distributed processes
        process_index = getattr(trainer.accelerator, "process_index", 0)
        
        # Create new environments for each batch of prompts
        # This simulates having separate environments for each rollout
        environments = []
        batch_indices = []
        for i in range(len(prompts) // args.num_generations):
            # Create a unique rollout ID for this batch
            batch_index = kwargs.get("batch_index", i)
            batch_indices.append(batch_index)
            rollout_id = f"{base_rollout_id}_{process_index}_{batch_index}"
            
            # Create a new environment with this rollout ID
            env = env_creator.create_env(f"{process_index}_{batch_index}")
            environments.append(env)
        
        # Split prompts into batches and process with separate environments
        results = {'ids': [], 'messages': [], 'mask': []}
        
        for i, env in enumerate(environments):
            # Get the slice of prompts for this environment
            start_idx = i * args.num_generations
            end_idx = start_idx + args.num_generations
            batch_prompts = prompts[start_idx:end_idx]
            
            # Generate completions using this environment
            batch_kwargs = dict(kwargs)
            batch_kwargs['batch_index'] = batch_indices[i]
            batch_kwargs['rollout_id'] = env.rollout_id
            env_result = env.generate(batch_prompts, llm, sampling_params, **batch_kwargs)
            
            # Append results
            results['ids'].extend(env_result['ids'])
            results['messages'].extend(env_result['messages'])
            results['mask'].extend(env_result['mask'])
        
        return results
    
    # Replace the original generate method with our wrapped version
    trainer.env.generate = wrapped_env_generate
    
    print("Starting GRPO training with isolated memory directories per rollout...")
    trainer.train()
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 