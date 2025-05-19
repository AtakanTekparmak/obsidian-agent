import os
import shutil
import json
from datetime import datetime

from agent.settings import (
    SYSTEM_PROMPT_PATH, FILE_SIZE_LIMIT, DIR_SIZE_LIMIT, 
    MEMORY_SIZE_LIMIT, MEMORY_PATH, LOG_DIR, REWARD_LOG_DIR, 
    COMPLETION_LOG_DIR, get_rollout_memory_path
)

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
    
def create_memory_if_not_exists(rollout_id=None) -> str:
    """
    Create the memory if it doesn't exist and return its path.
    
    Args:
        rollout_id: Optional rollout ID to create a unique memory directory
        
    Returns:
        The path to the memory directory
    """
    memory_path = get_rollout_memory_path(rollout_id)
    if not os.path.exists(memory_path):
        os.makedirs(memory_path, exist_ok=True)
    return memory_path

def create_log_dirs():
    """
    Create log directories if they don't exist.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REWARD_LOG_DIR, exist_ok=True)
    os.makedirs(COMPLETION_LOG_DIR, exist_ok=True)

def log_reward_calculation(persona_id, facts, memory_dump, reward, rollout_id=None):
    """
    Log a reward calculation to file.
    
    Args:
        persona_id: The ID of the persona
        facts: The facts to check
        memory_dump: The memory dump string
        reward: The calculated reward
        rollout_id: The rollout ID (optional)
    """
    create_log_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rollout_str = f"_rollout_{rollout_id}" if rollout_id is not None else ""
    log_file = os.path.join(REWARD_LOG_DIR, f"{timestamp}_{persona_id}{rollout_str}.json")
    
    log_data = {
        "timestamp": timestamp,
        "persona_id": persona_id,
        "rollout_id": rollout_id,
        "facts": facts,
        "reward": reward,
        "memory_dump_summary": memory_dump[:1000] + "..." if len(memory_dump) > 1000 else memory_dump
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)

def log_completion(persona_id, completion, rollout_id=None):
    """
    Log an agent completion to file.
    
    Args:
        persona_id: The ID of the persona
        completion: The completion from the agent
        rollout_id: The rollout ID (optional)
    """
    create_log_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rollout_str = f"_rollout_{rollout_id}" if rollout_id is not None else ""
    log_file = os.path.join(COMPLETION_LOG_DIR, f"{timestamp}_{persona_id}{rollout_str}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Persona ID: {persona_id}\n")
        f.write(f"Rollout ID: {rollout_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(completion)

def delete_memory(rollout_id=None) -> None:
    """
    Delete the memory.
    
    Args:
        rollout_id: Optional rollout ID to delete a specific memory directory
    """
    memory_path = get_rollout_memory_path(rollout_id)
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