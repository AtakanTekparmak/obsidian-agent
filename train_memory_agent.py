import os
import json
from pathlib import Path
from datasets import Dataset
import argparse

from verifiers.envs.memory_env import ObsidianAgentEnv
from verifiers.trainers import GRPOEnvTrainer
from verifiers.utils.data_utils import preprocess_dataset, format_dataset
from trl import GRPOConfig


# GPU setup for 8 GPUs (e.g., 2 for vLLM inference, 6 for training)
# This is an example, adjust gpu allocation and model paths as needed.
"""
8 GPU setup
# Terminal 1: Start vLLM server 

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
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

    # Initialize the environment to access its system_prompt and few_shot_examples
    # This instance is just for configuration; the trainer will manage its own env instance.
    env_for_formatting = ObsidianAgentEnv()

    # Format datasets to include the 'prompt' field expected by the trainer
    print("Formatting datasets...")
    train_ds = format_dataset(
        train_ds,
        system_prompt=env_for_formatting.system_prompt,
        question_key="question", # 'question' is the key from preprocess_convos_persona
        answer_key="answer"      # 'answer' is also present
    )
    if eval_ds:
        eval_ds = format_dataset(
            eval_ds,
            system_prompt=env_for_formatting.system_prompt,
            question_key="question",
            answer_key="answer"
        )

    # Initialize the environment
    # ObsidianAgentEnv does not need dataset paths in __init__ anymore.
    # It will receive data points from the trainer via env.reset(data=...)
    env = ObsidianAgentEnv(
        # system_prompt=MEMORY_AGENT_PROMPT, # Uses its own default MEMORY_AGENT_PROMPT
        # few_shot=[], # Add if needed
        # sampling_args, max_steps, etc. can be configured if defaults are not suitable
        num_generations=args.num_generations
    )

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
    
    # Initialize the trainer
    # The GRPOEnvTrainer expects the model argument to be the model name/path for HF models,
    # or a pre-loaded model object.
    trainer = GRPOEnvTrainer(
        model=args.model_name, 
        # processing_class can be specified if custom tokenizer/processing is needed
        env=env,
        # reward_funcs and reward_weights are fetched from the environment
        reward_funcs=env.get_reward_funcs(),
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    
    print("Starting GRPO training...")
    trainer.train()
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 