#!/usr/bin/env python3
"""
Main entrypoint for Obsidian Retrieval training with SkyRL.
Based on SkyRL examples but adapted for our retrieval environment.
"""

import ray
from omegaconf import DictConfig, OmegaConf
import sys
import os

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

class RetrievalPPOExp(BasePPOExp):
    """Custom experiment class for retrieval training."""
    
    def __init__(self, cfg: DictConfig):
        # Register the obsidian-retrieval environment
        register(
            id="obsidian-retrieval",
            entry_point="training.retrieval.env:RetrievalEnv",
        )
        super().__init__(cfg)

@ray.remote(num_cpus=1)
def main_remote(cfg: DictConfig):
    """Remote main function to run training."""
    exp = RetrievalPPOExp(cfg)
    exp.run()

def main():
    """Main function that parses arguments and launches training."""
    # Parse command line arguments as key=value pairs
    overrides = sys.argv[1:]
    
    # Calculate obsidian_root path early so we can use it in base_config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obsidian_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Create base configuration with sensible defaults
    base_config = {
        # Environment settings
        "environment": {
            "env_class": "obsidian-retrieval",
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
                    "path": "Qwen/Qwen3-4B",
                    "trust_remote_code": True
                },
                "optimizer_config": {
                    "lr": 1.0e-6
                }
            },
            "placement": {
                "colocate_all": True,
                "policy_num_gpus_per_node": 2
            },
            "algorithm": {
                "advantage_estimator": "grpo",
                "use_kl_loss": False
            },
            "epochs": 2,
            "train_batch_size": 64,
            "policy_mini_batch_size": 32,
            "micro_forward_batch_size_per_gpu": 4,
            "micro_train_batch_size_per_gpu": 2,
            "eval_batch_size": 32,
            "eval_before_train": True,
            "eval_interval": 1,
            "max_prompt_length": 8192,
            "logger": "console",
            "project_name": "obsidian-retrieval-skyrl",
            "run_name": "obsidian-retrieval-qwen3-4b",
            "output_dir": "./output/training/obsidian-retrieval-qwen3-4b",
            "ckpt_path": "./output/training/obsidian-retrieval-qwen3-4b/ckpt",
            "export_path": "./output/training/obsidian-retrieval-qwen3-4b/export",
            "save_strategy": "epoch",
            "save_total_limit": 4,
            "disable_fast_tokenizer": False
        },
        
        # Generator settings
        "generator": {
            "backend": "vllm",
            "num_inference_engines": 2,
            "inference_engine_tensor_parallel_size": 2,
            "n_samples_per_prompt": 4,
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_generate_length": 2048
            },
            "max_input_length": 8192,
            "async_engine": False,
            "batched": True,
            "max_turns": 8
        }
    }
    
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
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize Ray and run training
    initialize_ray(cfg)
    ray.get(main_remote.remote(cfg))

if __name__ == "__main__":
    main() 