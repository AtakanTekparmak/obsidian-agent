import os

def should_ignore(name):
    """Check if a file/directory should be ignored based on simple patterns."""
    ignore_patterns = {
        '.git', '.DS_Store', 'Thumbs.db', '__pycache__',
        '.pytest_cache', '.mypy_cache', 'node_modules',
        '.env', '.venv', 'venv', '.idea', '.vscode'
    }
    return name in ignore_patterns or name.startswith('.')

def generate_tree(directory, prefix='', is_last=True):
    """Generate directory tree structure."""
    result = []
    
    dir_name = os.path.basename(directory)
    
    # Add current directory to output
    if prefix == '':
        result.append(f"{dir_name}/")
    else:
        connector = '└── ' if is_last else '├── '
        result.append(f"{prefix}{connector}{dir_name}/")
    
    try:
        # Get all items and sort them (directories first, then files)
        all_items = os.listdir(directory)
        all_items = [item for item in all_items if not should_ignore(item)]
        all_items.sort(key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x.lower()))
        
        for i, item in enumerate(all_items):
            item_path = os.path.join(directory, item)
            is_last_child = i == len(all_items) - 1
            
            if os.path.isdir(item_path):
                new_prefix = prefix + ('    ' if is_last else '│   ')
                result.extend(generate_tree(item_path, new_prefix, is_last_child))
            else:
                connector = '└── ' if is_last_child else '├── '
                result.append(f"{prefix}{'    ' if is_last else '│   '}{connector}{item}")
    
    except PermissionError:
        pass
    
    return result

def get_file_contents(directory):
    """Get all file contents from directory recursively."""
    contents = {}
    
    for root, dirs, files in os.walk(directory):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(d)]
        
        # Filter out ignored files
        files = [f for f in files if not should_ignore(f)]
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    relative_path = os.path.relpath(file_path, directory)
                    contents[relative_path] = f.read()
            except UnicodeDecodeError:
                relative_path = os.path.relpath(file_path, directory)
                contents[relative_path] = "Skipped: Binary/non-text file"
            except Exception as e:
                relative_path = os.path.relpath(file_path, directory)
                contents[relative_path] = f"Skipped: {str(e)}"
    
    return contents

def dump_folder(folder_path: str):
    """
    Dump the contents of a folder to a string.
    Works with any folder, no git functionality required.
    """
    # Clean up path
    folder_path = os.path.abspath(folder_path.rstrip(os.sep))
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path {folder_path} is not a directory")
    
    # Generate directory tree
    tree = generate_tree(folder_path)
    
    # Get file contents
    file_contents = get_file_contents(folder_path)
    
    # Build output string
    output = "DIRECTORY STRUCTURE:\n"
    if tree:
        output += '\n'.join(tree)
    else:
        output += "(empty directory)"
    
    output += "\n\nFILE CONTENTS:\n\n"
    
    if file_contents:
        for path, content in sorted(file_contents.items()):
            output += f"════════ {path} ════════\n"
            output += content + "\n\n"
            output += "-"*80 + "\n\n"
    else:
        output += "(no files found)\n\n"

    return output