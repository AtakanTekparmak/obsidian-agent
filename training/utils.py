import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TrainingLogger:
    """Logger for training processes with rollout-specific logs"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.logger = logging.getLogger("training")
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create subdirectories
        self.reward_log_dir = os.path.join(log_dir, "rewards")
        self.completion_log_dir = os.path.join(log_dir, "completions")
        
        if not os.path.exists(self.reward_log_dir):
            os.makedirs(self.reward_log_dir)
        if not os.path.exists(self.completion_log_dir):
            os.makedirs(self.completion_log_dir)
    
    def log_reward_call(self, 
                       rollout_id: int, 
                       memory_dump: str, 
                       facts_to_check: List[Dict[str, Any]], 
                       reward_value: float) -> None:
        """Log a reward calculation call"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(self.reward_log_dir, f"reward_rollout_{rollout_id}_{timestamp}.json")
        
        data = {
            "rollout_id": rollout_id,
            "memory_dump": memory_dump,
            "facts_to_check": facts_to_check,
            "reward_value": reward_value,
            "timestamp": timestamp
        }
        
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Logged reward call for rollout {rollout_id} with value {reward_value}")
    
    def log_completion(self, 
                      rollout_id: int, 
                      prompt: Dict[str, Any], 
                      completion: Dict[str, Any],
                      step: Optional[int] = None) -> None:
        """Log a model completion"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        step_str = f"_step_{step}" if step is not None else ""
        log_file = os.path.join(self.completion_log_dir, f"completion_rollout_{rollout_id}{step_str}_{timestamp}.json")
        
        data = {
            "rollout_id": rollout_id,
            "prompt": prompt,
            "completion": completion,
            "step": step,
            "timestamp": timestamp
        }
        
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Logged completion for rollout {rollout_id}")

# Global logger instance
trainer_logger = TrainingLogger() 