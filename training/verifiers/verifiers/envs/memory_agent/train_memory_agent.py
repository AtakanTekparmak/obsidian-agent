import os
import json
from pathlib import Path
from datasets import Dataset
import argparse

from verifiers.envs.memory_agent.obsidian_agent_env import ObsidianAgentEnv
from verifiers.trainers import GRPOEnvTrainer
from verifiers.utils.data_utils import load_memory_dataset, format_dataset
from trl import GRPOConfig

def format_prompt(user_message, facts=None):
    """
    Format a user message into a prompt suitable for the LLM.
    """
    if facts:
        facts_str = "\n".join([f"- {fact['fact_description_or_change']}" for fact in facts])
        return f"User: {user_message}\n\nRelevant facts to remember:\n{facts_str}"
    return f"User: {user_message}"

def load_convo_data(convo_file_path):
    """
    Load conversation data from the JSON file.
    
    Args:
        convo_file_path: Path to the JSON file containing conversations
        
    Returns:
        A list of dictionaries containing the conversation data
    """
    with open(convo_file_path, "r") as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    
    for convo in data.get("convos", []):
        persona_name = convo.get("user_persona_name_surname", "")
        chats = convo.get("user_chats", [])
        
        for chat in chats:
            turn_number = chat.get("turn_number", 0)
            base_fact = chat.get("base_fact", "")
            user_message = chat.get("user_message", "")
            facts_to_check = chat.get("facts_to_check_so_far", [])
            
            # Format the prompt according to the verifiers framework requirements
            formatted_prompt = format_prompt(user_message, facts_to_check)
            
            # Create training example compatible with verifiers
            formatted_data.append({
                "question": formatted_prompt,  # This becomes the input to the model
                "answer": "",  # We don't have a specific answer string to check against
                "task": "memory",  # Custom task type
                "persona_name": persona_name,
                "turn_number": turn_number,
                "base_fact": base_fact,
                "user_message": user_message,
                "facts_to_check_so_far": facts_to_check,
            })
    
    return formatted_data

def format_dataset_for_training(dataset):
    """
    Format the dataset for training with GRPO.
    
    Args:
        dataset: The dataset to format
        
    Returns:
        The formatted dataset
    """
    def format_example(example):
        # Create chat messages format
        messages = [
            {"role": "user", "content": example["question"]}
        ]
        
        return {
            "prompt": messages,
            "facts_to_check_so_far": example["facts_to_check_so_far"],
            "answer": example["answer"],
            "task": example["task"]
        }
    
    return dataset.map(format_example)

def main():
    parser = argparse.ArgumentParser(description="Train Memory Agent")
    parser.add_argument("--data_path", type=str, default="data/convos.json", 
                        help="Path to conversation data JSON file")
    parser.add_argument("--output_dir", type=str, default="results/memory_agent", 
                        help="Directory to save model and results")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs for training")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", 
                        help="Model to use for training")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting")
    parser.add_argument("--num_generations", type=int, default=2,
                        help="Number of generations per prompt (must be divisible by batch size)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the memory dataset
    data_path = Path(args.data_path).resolve()
    print(f"Loading data from {data_path}")
    
    # Load and preprocess the dataset using the utility function
    dataset = load_memory_dataset(
        file_path=data_path,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Split into train and test
    splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    
    # Format the datasets for training (add system prompt and format as chat)
    train_dataset = format_dataset(
        splits["train"],
        system_prompt=None,  # System prompt is handled by the environment
    )
    
    eval_dataset = format_dataset(
        splits["test"],
        system_prompt=None,  # System prompt is handled by the environment
    )
    
    # Initialize the environment
    env = ObsidianAgentEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Configure GRPO training
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        use_vllm=True,  # GRPO requires vLLM
        temperature=0.7,
        top_p=0.9,
        num_generations=args.num_generations,  # Set generations to match batch size
    )
    
    # Initialize the trainer
    trainer = GRPOEnvTrainer(
        model=args.model_name,
        env=env,
        reward_funcs=env.get_reward_funcs(),
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train the model
    trainer.train()
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 