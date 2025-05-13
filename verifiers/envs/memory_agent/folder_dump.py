import os
import sys
import subprocess

def is_git_repo(path):
    """Check if the path is within a Git repository."""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'],
                       cwd=path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def get_git_root(path):
    """Get the root directory of the Git repository."""
    try:
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                                cwd=path, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def is_ignored(path, git_root):
    """Check if a path is ignored by Git."""
    if '.git/' in path.replace(os.sep, '/'):
        return True  # Always ignore .git directory
    
    if not git_root:
        return False
    
    try:
        rel_path = os.path.relpath(path, git_root)
        result = subprocess.run(['git', 'check-ignore', '--quiet', rel_path],
                                cwd=git_root, check=False)
        return result.returncode == 0
    except Exception:
        return False

def generate_tree(directory, prefix='', is_last=True, git_root=None, output=None):
    """Generate directory tree structure with Git ignore support."""
    if output is None:
        output = []
    
    dir_name = os.path.basename(directory)
    if dir_name == '.git':
        return output
    
    if is_ignored(directory, git_root):
        return output
    
    # Add current directory to output
    if prefix == '':
        output.append(f"{dir_name}/")
    else:
        connector = '└── ' if is_last else '├── '
        output.append(f"{prefix}{connector}{dir_name}/")
    
    # Get sorted list of children
    try:
        children = sorted(os.listdir(directory), key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x))
    except PermissionError:
        return output
    
    # Filter ignored paths
    children = [c for c in children if not is_ignored(os.path.join(directory, c), git_root)]
    
    for i, child in enumerate(children):
        child_path = os.path.join(directory, child)
        is_last_child = i == len(children) - 1
        
        if os.path.isdir(child_path):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            generate_tree(child_path, new_prefix, is_last_child, git_root, output)
        else:
            connector = '└── ' if is_last_child else '├── '
            output.append(f"{prefix}{connector}{child}")
    
    return output

def get_file_contents(directory, git_root):
    """Collect non-ignored file contents with error handling."""
    contents = {}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), git_root)]
        files = [f for f in files if not is_ignored(os.path.join(root, f), git_root)]
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    contents[os.path.relpath(file_path, directory)] = f.read()
            except UnicodeDecodeError:
                contents[os.path.relpath(file_path, directory)] = "Skipped: Binary/non-text file"
            except Exception as e:
                contents[os.path.relpath(file_path, directory)] = f"Skipped: {str(e)}"
    return contents

def main():
    # Set input directory
    input_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    input_dir = os.path.abspath(input_dir)
    
    # Git configuration
    git_root = get_git_root(input_dir) if is_git_repo(input_dir) else None
    
    # Generate directory tree
    tree = generate_tree(input_dir, git_root=git_root)
    
    # Get file contents
    file_contents = get_file_contents(input_dir, git_root)
    
    # Write output
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("DIRECTORY STRUCTURE:\n")
        f.write('\n'.join(tree))
        f.write("\n\nFILE CONTENTS:\n\n")
        for path, content in file_contents.items():
            f.write(f"════════ {path} ════════\n")
            f.write(content + "\n\n")

if __name__ == "__main__":
    main()