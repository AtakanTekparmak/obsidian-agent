#!/usr/bin/env python
import os
import shutil
from verifiers.envs.memory_env import ObsidianAgentEnv, MEMORY_DIRS_ROOT

# Clean up any existing memory dirs
if os.path.exists(MEMORY_DIRS_ROOT):
    shutil.rmtree(MEMORY_DIRS_ROOT)
os.makedirs(MEMORY_DIRS_ROOT)

# Create multiple environment instances to simulate rollouts
print("Creating multiple environment instances...")
envs = []
for i in range(3):  # Create 3 instances
    env = ObsidianAgentEnv(
        convos_dataset_path="training/data/convos.json"
    )
    envs.append(env)
    print(f"Created env with ID {env.instance_id}, memory dir: {env.memory_dir}")

# Test that each environment has a unique memory directory
memory_dirs = [env.memory_dir for env in envs]
print(f"Memory directories: {memory_dirs}")
assert len(set(memory_dirs)) == len(memory_dirs), "Memory directories are not unique!"
print("All memory directories are unique ✓")

# Test that each environment can write to its memory directory
for i, env in enumerate(envs):
    test_file = os.path.join(env.memory_dir, f"test_file_{i}.txt")
    with open(test_file, "w") as f:
        f.write(f"Test content for env {env.instance_id}")
    print(f"Created test file in env {env.instance_id}: {test_file}")

# Verify files exist in correct directories
for i, env in enumerate(envs):
    test_file = os.path.join(env.memory_dir, f"test_file_{i}.txt")
    assert os.path.exists(test_file), f"Test file not found in {test_file}"
    print(f"Found test file for env {env.instance_id} ✓")

# Verify each directory only contains its own files
for i, env in enumerate(envs):
    files = os.listdir(env.memory_dir)
    assert len(files) == 1, f"Expected 1 file in {env.memory_dir}, but found {len(files)}: {files}"
    assert files[0] == f"test_file_{i}.txt", f"Found unexpected file in {env.memory_dir}: {files}"
    print(f"Directory for env {env.instance_id} contains only its own file ✓")

print("\nAll tests passed! The memory directory implementation is working correctly.") 