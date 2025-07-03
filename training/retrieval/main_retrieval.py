#!/usr/bin/env python3
"""
Main entrypoint for Obsidian Retrieval training with SkyRL.
Based on SkyRL examples but adapted for our retrieval environment.
"""

import ray
from omegaconf import DictConfig, OmegaConf
import sys
import os
import argparse

# Add obsidian-agent root directory to Python path if running from SkyRL directory
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
obsidian_root = os.path.join(script_dir, "..", "..")
obsidian_root = os.path.abspath(obsidian_root)

if obsidian_root not in sys.path:
    sys.path.insert(0, obsidian_root)

from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_gym.envs import register
import skyrl_gym.error

# Register the environment at module level (only once)
try:
    register(
        id="obsidian-retrieval-env",
        entry_point="training.retrieval.new_env:ObsidianRetrievalEnv",
    )
    print("Successfully registered obsidian-retrieval environment")
except skyrl_gym.error.RegistrationError:
    print("obsidian-retrieval environment already registered, skipping registration")

class RetrievalPPOExp(BasePPOExp):
    """Custom experiment class for retrieval training."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

@ray.remote(num_cpus=1)
def main_remote(cfg: DictConfig):
    """Remote main function to run training."""
    import sys
    import os
    
    # Add obsidian-agent root directory to Python path in Ray worker context
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obsidian_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    if obsidian_root not in sys.path:
        sys.path.insert(0, obsidian_root)
    
    # Import modules to ensure they're available
    try:
        import training
        import agent
        print(f"Successfully imported training module from: {training.__file__}")
        print(f"Successfully imported agent module from: {agent.__file__}")
    except ImportError as e:
        print(f"Failed to import modules: {e}")
        print(f"Python path: {sys.path}")
        print(f"Obsidian root: {obsidian_root}")
        raise
    
    exp = RetrievalPPOExp(cfg)
    exp.run()

def main():
    """Main function that parses arguments and launches training."""
    # Parse --single-agent flag separately
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--single-agent', action='store_true', 
                        help='Run a single agent rollout for diagnostic purposes')
    
    # Parse known args (just --single-agent) and leave the rest as overrides
    args, remaining_args = parser.parse_known_args()
    
    # The remaining args are the key=value overrides for OmegaConf
    overrides = remaining_args
    
    # Calculate obsidian_root path early so we can use it in base_config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obsidian_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Create base configuration with sensible defaults
    base_config = {
        # Environment settings
        "environment": {
            "env_class": "obsidian-retrieval-env",
            "skyrl_gym": {
                "env_kwargs": {},
                "extras": {}
            }
        },
        
        # Data settings - use relative paths from script directory
        "data": {
            "train_data": [os.path.join(obsidian_root, "output", "datasets", "skyrl_formatted", "train.parquet")],
            "val_data": [os.path.join(obsidian_root, "output", "datasets", "skyrl_formatted", "validation.parquet")]
        },
        
        # Trainer settings
        "trainer": {
            "policy": {
                "model": {
                    "path": "Qwen/Qwen3-14B",
                    "trust_remote_code": True
                },
                "optimizer_config": {
                    "lr": 1.0e-6
                }
            },
            "placement": {
                "colocate_all": True,
                "policy_num_gpus_per_node": 4
            },
            "algorithm": {
                "advantage_estimator": "grpo",
                "use_kl_loss": False
            },
            "epochs": 1,
            "train_batch_size": 64,
            "policy_mini_batch_size": 32,
            "micro_forward_batch_size_per_gpu": 4,
            "micro_train_batch_size_per_gpu": 2,
            "eval_batch_size": 32,
            "eval_before_train": True,
            "eval_interval": 1,
            "max_prompt_length": 16384,
            "logger": "console",
            "project_name": "obsidian-retrieval-skyrl",
            "run_name": "obsidian-retrieval-qwen3-8b",
            "output_dir": "./output/training/obsidian-retrieval-qwen3-8b",
            "ckpt_path": "./output/training/obsidian-retrieval-qwen3-8b/ckpt",
            "export_path": "./output/training/obsidian-retrieval-qwen3-8b/export",
            "save_strategy": "epoch",
            "save_total_limit": 4,
            "disable_fast_tokenizer": False,
            "num_policy_gpus": 4,
            "num_rollout_gpus": 4
        },
        
        # Generator settings
        "generator": {
            "backend": "vllm",
            "num_inference_engines": 2,
            "inference_engine_tensor_parallel_size": 2,
            "n_samples_per_prompt": 1,
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_generate_length": 2048,
                "stop": ["</python>", "</reply>"]
            },
            "max_input_length": 16384,
            "async_engine": False,
            "batched": True,
            "max_turns": 8
        }
    }
    
    # If --single-agent flag is set, modify configuration for single agent diagnostic
    if args.single_agent:
        print("\n=== SINGLE AGENT MODE ENABLED ===")
        print("Running diagnostic mode with a single agent rollout\n")
        
        # Override settings for single agent diagnostic
        single_agent_overrides = {
            "trainer": {
                "train_batch_size": 1,
                "policy_mini_batch_size": 1,
                "micro_forward_batch_size_per_gpu": 1,
                "micro_train_batch_size_per_gpu": 1,
                "eval_batch_size": 1,
                "eval_before_train": False,  # Skip initial evaluation
                "eval_interval": 1, 
                "epochs": 1,
                "run_name": "obsidian-retrieval-single-agent-diagnostic",
                "output_dir": "./output/training/single-agent-diagnostic",
                "logger": "console",  # Ensure console logging for diagnostics
            },
            "generator": {
                "n_samples_per_prompt": 1,  # Only generate one sample
                "sampling_params": {
                    "temperature": 0.7,  # Keep the same sampling params
                    "top_p": 0.9,
                    "max_generate_length": 2048,
                    "stop": ["</python>", "</reply>"]  # Keep stop tokens
                }
            }
        }
        
        # Merge single agent overrides into base config
        base_config = OmegaConf.merge(OmegaConf.create(base_config), 
                                      OmegaConf.create(single_agent_overrides))
        base_config = OmegaConf.to_container(base_config)
    
    # Load SkyRL's default PPO configuration. This ensures that all
    # required fields expected by ``skyrl_train`` are present.
    cwd = os.getcwd()
    
    # When running from SkyRL/skyrl-train directory, the config is in the current directory
    if "SkyRL/skyrl-train" in cwd:
        skyrl_base_cfg_path = os.path.join("skyrl_train", "config", "ppo_base_config.yaml")
        # Use absolute path to deepspeed config
        deepspeed_cfg_path = os.path.join(obsidian_root, "training", "configs", "rl", "deepspeed_zero3.json")
    else:
        # Fallback to original paths for when running from obsidian-agent root
        skyrl_base_cfg_path = os.path.join(obsidian_root, "SkyRL", "skyrl-train", "skyrl_train", "config", "ppo_base_config.yaml")
        deepspeed_cfg_path = os.path.join(obsidian_root, "training", "configs", "rl", "deepspeed_zero3.json")
    
    skyrl_base_cfg = OmegaConf.load(skyrl_base_cfg_path)

    # Load the deepspeed config
    deepspeed_cfg = OmegaConf.load(deepspeed_cfg_path)

    # Create OmegaConf config from our base and merge with SkyRL defaults
    cfg = OmegaConf.merge(skyrl_base_cfg, OmegaConf.create(base_config))
    cfg.deepspeed_config = {"train": deepspeed_cfg, "eval": deepspeed_cfg}
    OmegaConf.resolve(cfg)
    
    # Apply command line overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    if args.single_agent:
        print("Final configuration for single agent mode:")
        print(f"  train_batch_size: {cfg.trainer.train_batch_size}")
        print(f"  n_samples_per_prompt: {cfg.generator.n_samples_per_prompt}")
        print(f"  eval_before_train: {cfg.trainer.eval_before_train}")
        print(f"  epochs: {cfg.trainer.epochs}")
        print(f"  output_dir: {cfg.trainer.output_dir}")
        print()
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize Ray and run training
    initialize_ray(cfg)
    ray.get(main_remote.remote(cfg))

if __name__ == "__main__":
    main() 