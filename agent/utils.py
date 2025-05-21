import os
import shutil

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
    
def create_memory_if_not_exists(path: str = MEMORY_PATH):
    """
    Create the memory if it doesn't exist.
    
    Args:
        path: The path to create. Defaults to MEMORY_PATH.
        
    Returns:
        None
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating memory directory at {path}: {e}")

def delete_memory() -> None:
    """
    Delete the memory.
    """
    if os.path.exists(MEMORY_PATH):
        shutil.rmtree(MEMORY_PATH)

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
    return "<result>\n" + str(results) + "\n</result>"