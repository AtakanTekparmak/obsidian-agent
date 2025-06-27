from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset


def load_skyrl_dataset(json_path: str) -> List[Dict]:
    """
    Load the SkyRL dataset from JSON file.
    
    Args:
        json_path: Path to the JSON dataset file
        
    Returns:
        List of dataset entries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def validate_dataset_entry(entry: Dict) -> bool:
    """
    Validate that a dataset entry has all required SkyRL fields.
    
    Args:
        entry: A single dataset entry
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['data_source', 'prompt', 'env_class', 'reward_spec']
    
    for field in required_fields:
        if field not in entry:
            print(f"Missing required field: {field}")
            return False
    
    # Validate prompt structure
    if not isinstance(entry['prompt'], list):
        print("prompt field must be a list")
        return False
    
    for message in entry['prompt']:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            print("Each prompt message must have 'role' and 'content' fields")
            return False
    
    # Validate reward_spec structure
    reward_spec = entry['reward_spec']
    if not isinstance(reward_spec, dict):
        print("reward_spec must be a dictionary")
        return False
    
    if 'method' not in reward_spec:
        print("reward_spec must have 'method' field")
        return False
    
    if reward_spec['method'] not in ['rule', 'reward_model']:
        print("reward_spec method must be either 'rule' or 'reward_model'")
        return False
    
    if reward_spec['method'] == 'rule' and 'ground_truth' not in reward_spec:
        print("reward_spec with method 'rule' must have 'ground_truth' field")
        return False
    
    return True


def format_dataset_for_skyrl(dataset: List[Dict]) -> List[Dict]:
    """
    Format the dataset entries to ensure they conform to SkyRL requirements.
    
    Args:
        dataset: List of dataset entries
        
    Returns:
        Formatted dataset entries
    """
    formatted_dataset = []
    
    for i, entry in enumerate(dataset):
        if not validate_dataset_entry(entry):
            print(f"Skipping invalid entry at index {i}")
            continue
        
        # The dataset from generate_kb.py already follows the correct format,
        # but we'll ensure all fields are present and properly structured
        formatted_entry = {
            "data_source": entry["data_source"],
            "prompt": entry["prompt"],
            "env_class": entry["env_class"],
            "reward_spec": entry["reward_spec"],
        }
        
        # Add extra_info if present
        if "extra_info" in entry:
            formatted_entry["extra_info"] = entry["extra_info"]
        
        formatted_dataset.append(formatted_entry)
    
    return formatted_dataset


def split_dataset(dataset: List[Dict], train_ratio: float = 0.8) -> tuple[List[Dict], List[Dict]]:
    """
    Split the dataset into training and validation sets.
    
    Args:
        dataset: List of dataset entries
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    split_idx = int(len(dataset) * train_ratio)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    
    return train_dataset, val_dataset


def save_to_parquet(dataset: List[Dict], output_path: str) -> None:
    """
    Save dataset to parquet format using HuggingFace datasets.
    
    Args:
        dataset: List of dataset entries
        output_path: Path to save the parquet file
    """
    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset)
    
    # Save as parquet
    hf_dataset.to_parquet(output_path)
    print(f"Saved {len(dataset)} entries to {output_path}")


def main():
    """Main function to format and convert SkyRL dataset."""
    parser = argparse.ArgumentParser(
        description="Format SkyRL dataset from JSON to parquet format"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="output/datasets/skrl_dataset.json",
        help="Path to input JSON dataset file (default: output/datasets/skrl_dataset.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/datasets/skyrl_formatted",
        help="Output directory for parquet files (default: output/datasets/skyrl_formatted)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Don't split into train/validation, save as single file"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_json):
        print(f"Error: Input file {args.input_json} does not exist")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and format dataset
    print(f"Loading dataset from {args.input_json}")
    dataset = load_skyrl_dataset(args.input_json)
    print(f"Loaded {len(dataset)} entries")
    
    # Format dataset
    print("Formatting dataset for SkyRL...")
    formatted_dataset = format_dataset_for_skyrl(dataset)
    print(f"Formatted {len(formatted_dataset)} valid entries")
    
    if len(formatted_dataset) == 0:
        print("No valid entries found. Exiting.")
        return
    
    # Save dataset
    if args.no_split:
        # Save as single file
        output_path = output_dir / "dataset.parquet"
        save_to_parquet(formatted_dataset, str(output_path))
    else:
        # Split and save
        train_dataset, val_dataset = split_dataset(formatted_dataset, args.train_ratio)
        
        print(f"Split into {len(train_dataset)} training and {len(val_dataset)} validation entries")
        
        # Save training set
        train_path = output_dir / "train.parquet"
        save_to_parquet(train_dataset, str(train_path))
        
        # Save validation set
        val_path = output_dir / "validation.parquet"
        save_to_parquet(val_dataset, str(val_path))
    
    print("Dataset formatting complete!")
    
    # Print sample entry for verification
    if formatted_dataset:
        print("\nSample dataset entry:")
        sample_entry = formatted_dataset[0]
        print(f"  data_source: {sample_entry['data_source']}")
        print(f"  env_class: {sample_entry['env_class']}")
        print(f"  prompt: {len(sample_entry['prompt'])} messages")
        print(f"  reward_spec method: {sample_entry['reward_spec']['method']}")
        if 'extra_info' in sample_entry:
            extra_keys = list(sample_entry['extra_info'].keys())
            print(f"  extra_info keys: {extra_keys}")


if __name__ == "__main__":
    main() 