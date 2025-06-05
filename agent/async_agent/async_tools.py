import os
import aiofiles
import aiofiles.os
from typing import List, Union

from agent.settings import MEMORY_PATH
from agent.utils import check_size_limits, create_memory_if_not_exists

async def get_size(file_or_dir_path: str) -> int:
    """
    Get the size of a file or directory asynchronously.

    Args:
        file_or_dir_path: The path to the file or directory.

    Returns:
        The size of the file or directory in bytes.
    """
    try:
        stat_result = await aiofiles.os.stat(file_or_dir_path)
        return stat_result.st_size
    except Exception as e:
        return 0

async def create_file(file_path: str, content: str = "") -> bool:
    """
    Create a new file in the memory with the given content (if any) asynchronously.
    First create a temporary file with the given content, check if 
    the size limits are respected, if so, move the temporary file to 
    the final destination.

    Args:
        file_path: The path to the file.
        content: The content of the file.
    """
    try:
        temp_file_path = "temp.txt"
        async with aiofiles.open(temp_file_path, "w") as f:
            await f.write(content)
        
        if check_size_limits(temp_file_path):
            # Move the temporary file to the final destination
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)
            await aiofiles.os.remove(temp_file_path)
            return True
        else:
            await aiofiles.os.remove(temp_file_path)
            return False
    except Exception as e:
        return f"Error: {e}"
    
async def create_dir(dir_path: str) -> bool:
    """
    Create a new directory in the memory asynchronously.

    Args:
        dir_path: The path to the directory.

    Returns:
        True if the directory was created successfully, False otherwise.
    """
    try:
        await aiofiles.os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as _:
        return False
    
async def write_to_file(file_path: str, content: str) -> bool:
    """
    Write to a file in the memory asynchronously. First create a temporary file with 
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
        try:
            if await aiofiles.os.path.exists(file_path):
                async with aiofiles.open(file_path, "r") as f:
                    original_content = await f.read()
        except:
            original_content = ""
            
        temp_file_path = "temp.txt"
        async with aiofiles.open(temp_file_path, "w") as f:
            await f.write(original_content + "\n" + content)
            
        if check_size_limits(temp_file_path):
            await aiofiles.os.rename(temp_file_path, file_path)
            return True
        else:
            await aiofiles.os.remove(temp_file_path)
            return False
    except Exception as e:
        return f"Error: {e}"
    
async def read_file(file_path: str) -> str:
    """
    Read a file in the memory asynchronously.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file.
    """
    try:
        async with aiofiles.open(file_path, "r") as f:
            return await f.read()
    except Exception as e:
        return f"Error: {e}"
    
async def list_files(dir_path: str = None) -> List[str]:
    """
    List all files and directories in the memory asynchronously. Full paths 
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
            
        # Get the current working directory to use as base for relative paths
        cwd = os.getcwd()
            
        result_files = []
        
        # Since aiofiles doesn't have os.walk, we'll use a recursive approach
        async def _walk_directory(path: str):
            try:
                entries = await aiofiles.os.listdir(path)
                for entry in entries:
                    full_path = os.path.join(path, entry)
                    if await aiofiles.os.path.isfile(full_path):
                        try:
                            rel_path = os.path.relpath(full_path, cwd)
                            result_files.append(rel_path)
                        except ValueError:
                            result_files.append(full_path)
                    elif await aiofiles.os.path.isdir(full_path):
                        await _walk_directory(full_path)
            except Exception:
                pass
                
        await _walk_directory(dir_path)
        return result_files
    except Exception as e:
        return [f"Error: {e}"]
    
async def delete_file(file_path: str) -> bool:
    """
    Delete a file in the memory asynchronously.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file was deleted successfully, False otherwise. 
    """
    try:
        await aiofiles.os.remove(file_path)
        return True
    except Exception as _:
        return False
    
async def go_to_link(link_string: str) -> str:
    """
    Go to a link in the memory and return the content of the note Y asynchronously. A link in a note X to a note Y, with the
    path path/to/note/Y.md, is structured like this:
    [[path/to/note/Y]]

    Args:
        link_string: The link to go to.

    Returns:
        The content of the note Y.
    """
    try:
        file_path = link_string
        async with aiofiles.open(file_path, "r") as f:
            return await f.read()
    except Exception as _:
        return "Error: File not found"

async def check_if_file_exists(file_path: str) -> bool:
    """
    Check if a file exists in the given filepath asynchronously.
    """
    return await aiofiles.os.path.exists(file_path)

async def check_if_dir_exists(dir_path: str) -> bool:
    """
    Check if a directory exists in the given filepath asynchronously.
    """
    return await aiofiles.os.path.exists(dir_path) and await aiofiles.os.path.isdir(dir_path) 