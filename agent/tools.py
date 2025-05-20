import os

from agent.settings import MEMORY_PATH
from agent.utils import check_size_limits, create_memory_if_not_exists

def get_size(file_or_dir_path: str) -> int:
    """
    Get the size of a file or directory.

    Args:
        file_or_dir_path: The path to the file or directory.

    Returns:
        The size of the file or directory in bytes.
    """
    return os.path.getsize(file_or_dir_path)

def create_file(file_path: str, content: str = "") -> bool:
    """
    Create a new file in the memory with the given content (if any).
    First create a temporary file with the given content, check if 
    the size limits are respected, if so, move the temporary file to 
    the final destination.

    Args:
        file_path: The path to the file.
        content: The content of the file.
    """
    try:
        temp_file_path = "temp.txt"
        with open(temp_file_path, "w") as f:
            f.write(content)
        if check_size_limits(temp_file_path):
            # Move the temporary file to the final destination
            with open(file_path, "w") as f:
                f.write(content)
            os.remove(temp_file_path)
            return True
        else:
            os.remove(temp_file_path)
            return False
    except Exception as e:
        return f"Error: {e}"
    
def create_dir(dir_path: str) -> bool:
    """
    Create a new directory in the memory.

    Args:
        dir_path: The path to the directory.

    Returns:
        True if the directory was created successfully, False otherwise.
    """
    try:
        os.makedirs(dir_path)
        return True
    except Exception as _:
        return False
    
def write_to_file(file_path: str, content: str) -> bool:
    """
    Write to a file in the memory. First create a temporary file with ]
    original file content + new content. Check if the size limits are respected,
    if so, move the temporary file to the final destination.

    Args:
        file_path: The path to the file.
        content: The content to write to the file.

    Returns:
        True if the content was written successfully, False otherwise.
    """
    try:
        original_content = ""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                original_content = f.read()
        temp_file_path = "temp.txt"
        with open(temp_file_path, "w") as f:
            f.write(original_content + "\n" + content)
        if check_size_limits(temp_file_path):
            os.rename(temp_file_path, file_path)
            return True
        else:
            os.remove(temp_file_path)
            return False
    except Exception as e:
        return f"Error: {e}"
    
def read_file(file_path: str) -> str:
    """
    Read a file in the memory.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file.
    """
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"
    
def list_files(dir_path: str = None) -> list[str]:
    """
    List all files and directories in the memory. Full paths 
    are returned and directories are searched recursively. An
    example of the output is:
    ["dir/a.txt", "dir/b.txt", "dir/subdir/c.txt", "d.txt"]

    Args:
        dir_path: The path to the directory. If None, uses the current working directory.

    Returns:
        A list of files and directories in the memory.
    """
    try:
        # Use current directory if dir_path is None
        if dir_path is None:
            dir_path = os.getcwd()
            
        result_files = []
        for root, _, files_list in os.walk(dir_path):
            for file in files_list:
                result_files.append(os.path.join(root, file))
        return [file.split(f"{MEMORY_PATH}/", 1)[1] if f"{MEMORY_PATH}/" in file else file for file in result_files]
    except Exception as e:
        return [f"Error: {e}"]
    
def delete_file(file_path: str) -> bool:
    """
    Delete a file in the memory.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file was deleted successfully, False otherwise. 
    """
    try:
        os.remove(file_path)
        return True
    except Exception as _:
        return False
    
def go_to_link(link_string: str) -> str:
    """
    Go to a link in the memory and return the content of the note Y. A link in a note X to a note Y, with the
    path path/to/note/Y.md, is structured like this:
    [[path/to/note/Y]]

    Args:
        link_string: The link to go to.

    Returns:
        The content of the note Y.
    """
    try:
        file_path = link_string
        with open(file_path, "r") as f:
            return f.read()
    except Exception as _:
        return "Error: File not found"

def check_if_file_exists(file_path: str) -> bool:
    """
    Check if a file exists in the given filepath.
    """
    return os.path.exists(file_path)

def check_if_dir_exists(dir_path: str) -> bool:
    """
    Check if a directory exists in the given filepath.
    """
    return os.path.exists(dir_path)

