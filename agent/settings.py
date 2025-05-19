import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_STRONG_MODEL = "google/gemini-2.5-pro-preview-03-25"

# Memory
MEMORY_PATH = "memory_dir"
MEMORY_BASE_DIR = "memory"
FILE_SIZE_LIMIT = 1024 * 1024 # 1MB
DIR_SIZE_LIMIT = 1024 * 1024 * 10 # 10MB
MEMORY_SIZE_LIMIT = 1024 * 1024 * 100 # 100MB

# Engine
SANDBOX_TIMEOUT = 20

# Path settings
SYSTEM_PROMPT_PATH = "agent/system_prompt.txt"

# Log settings
LOG_DIR = "logs"
REWARD_LOG_DIR = os.path.join(LOG_DIR, "rewards")

# Create necessary directories
os.makedirs(MEMORY_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REWARD_LOG_DIR, exist_ok=True)

def get_rollout_memory_path(rollout_id=None):
    """
    Get a memory path for a specific rollout.
    If rollout_id is not provided, returns the default memory path.
    
    Args:
        rollout_id: Unique ID for the rollout
        
    Returns:
        Path to the memory directory for this rollout
    """
    if rollout_id is None:
        return MEMORY_PATH
    
    return os.path.join(MEMORY_BASE_DIR, f"memory_dir_{rollout_id}")