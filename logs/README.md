# Per-Rollout Memory Isolation for GRPO Training

This directory contains logs from the GRPO training process for the Obsidian Memory Agent. Each training run creates a timestamped subdirectory with detailed logs of the memory dump and reward calculations for each rollout.

## Implementation Details

### Problem Solved

The original implementation had a critical issue where all rollouts in GRPO training were sharing the same memory directory. This caused interference between different rollouts, leading to inconsistent training behavior and unreliable reward calculations.

The fixed implementation:
1. Creates a separate memory directory for each rollout
2. Ensures memory is properly reset between personas
3. Logs reward calculations for debugging and analysis

### Key Components

1. **ObsidianAgentEnv**: Modified to support customizable memory paths and per-rollout IDs
2. **MemoryRubric**: Updated to use the correct memory path for each rollout
3. **GRPOEnvTrainer**: Enhanced to create isolated environments for each rollout batch
4. **Logging**: Added comprehensive logging of memory dumps and reward calculations

## Log Directory Structure

Each training run creates a timestamped directory with the following structure:

```
logs/
├── 20230601_120000/
│   ├── rollout_1234abcd_0/
│   │   ├── memory_dump.txt    # Memory dump used for reward calculation
│   │   ├── facts.json         # Facts that were checked against the memory
│   │   └── reward.txt         # Final reward score
│   ├── rollout_5678efgh_0/
│   │   └── ...
│   └── ...
└── ...
```

## Usage

Examine these logs to:
1. Debug reward calculation issues
2. Analyze memory performance
3. Understand how different rollouts behave
4. Verify memory isolation is working correctly

## Improvements

This implementation eliminates cross-contamination between rollouts, allowing each trajectory to be evaluated independently. This is crucial for proper reinforcement learning since:

1. Each rollout should be a clean experiment starting from the same initial conditions
2. Reward calculations should only assess the performance of the specific agent that generated the data
3. Memory isolation ensures deterministic and reproducible behavior for each rollout 