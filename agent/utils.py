import os
import shutil
import json
import datetime
from pathlib import Path

from agent.settings import SYSTEM_PROMPT_PATH, FILE_SIZE_LIMIT, DIR_SIZE_LIMIT, MEMORY_SIZE_LIMIT, MEMORY_PATH

def load_system_prompt() -> str:
    """
    Load the system prompt from the file.

    Returns:
        The system prompt as a string.
    """
    try:
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt file not found at {SYSTEM_PROMPT_PATH}")
    
def check_file_size_limit(file_path: str) -> bool:
    """
    Check if the file size limit is respected.
    """
    return os.path.getsize(file_path) <= FILE_SIZE_LIMIT

def check_dir_size_limit(dir_path: str) -> bool:
    """
    Check if the directory size limit is respected.
    """
    return os.path.getsize(dir_path) <= DIR_SIZE_LIMIT

def check_memory_size_limit() -> bool:
    """
    Check if the memory size limit is respected.
    """
    current_working_dir = os.getcwd()
    return os.path.getsize(current_working_dir) <= MEMORY_SIZE_LIMIT

def check_size_limits(file_or_dir_path: str) -> bool:
    """
    Check if the size limits are respected.
    """
    if file_or_dir_path == "":
        return check_memory_size_limit()
    elif os.path.isdir(file_or_dir_path):
        return check_dir_size_limit(file_or_dir_path) and check_memory_size_limit()
    elif os.path.isfile(file_or_dir_path):
        parent_dir = os.path.dirname(file_or_dir_path)
        if not parent_dir == "":
            return check_file_size_limit(file_or_dir_path) and check_dir_size_limit(parent_dir) and check_memory_size_limit()
        else:
            return check_file_size_limit(file_or_dir_path) and check_memory_size_limit()
    else:
        return False
    
def create_memory_if_not_exists(memory_path: str = MEMORY_PATH) -> None:
    """
    Create the memory if it doesn't exist.
    
    Args:
        memory_path: Path to the memory directory
    """
    # Make sure parent directory exists first
    parent_dir = os.path.dirname(memory_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    if not os.path.exists(memory_path):
        os.makedirs(memory_path)

def create_log_dir() -> str:
    """
    Create a log directory for the current run if it doesn't exist.
    
    Returns:
        Path to the log directory
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir()
    
    # Create a subdirectory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = log_dir / timestamp
    if not run_log_dir.exists():
        run_log_dir.mkdir()
    
    return str(run_log_dir)

def log_reward_calculation(log_dir: str, rollout_id: int, memory_dump: str, facts: list, reward: float) -> None:
    """
    Log the reward calculation data.
    
    Args:
        log_dir: Path to the log directory
        rollout_id: ID of the rollout
        memory_dump: Memory dump string
        facts: Facts to check
        reward: Calculated reward
    """
    # Create rollout directory if it doesn't exist
    rollout_dir = Path(log_dir) / f"rollout_{rollout_id}"
    if not rollout_dir.exists():
        rollout_dir.mkdir()
    
    # Log memory dump
    with open(rollout_dir / "memory_dump.txt", "w") as f:
        f.write(memory_dump)
    
    # Log facts
    with open(rollout_dir / "facts.json", "w") as f:
        json.dump([fact.model_dump() for fact in facts], f, indent=2)
    
    # Log reward
    with open(rollout_dir / "reward.txt", "w") as f:
        f.write(str(reward))

def delete_memory(memory_path: str = MEMORY_PATH) -> None:
    """
    Delete the memory.
    
    Args:
        memory_path: Path to the memory directory to delete
    """
    if os.path.exists(memory_path):
        shutil.rmtree(memory_path)

def extract_python_code(response: str) -> str:
    """
    Extract the python code from the response.

    Args:
        response: The response from the model.

    Returns:
        The python code from the response.
    """
    if "```python" in response:
        return response.split("```python")[1].split("```")[0]
    else:
        return response

def format_results(results: dict) -> str:
    """
    Format the results into a string.
    """
    return "<r>\n" + str(results) + "\n</r>"