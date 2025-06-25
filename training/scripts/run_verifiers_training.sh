#!/bin/bash

# Script to run verifiers training with proper NCCL setup for cross-GPU communication
# This allows training on GPUs 4-7 to communicate with inference on GPUs 0-3

echo "Setting up environment for verifiers training..."

# Check if inference server is running
if ! curl -s http://0.0.0.0:8000/health/ > /dev/null; then
    echo "ERROR: vLLM inference server is not running on port 8000!"
    echo "Please start the inference server first with: make vf-inference"
    exit 1
fi

echo "Inference server is running. Starting training..."

# Set UV_FROZEN to true to avoid installing new packages
export UV_FROZEN=true 

# Set NCCL environment for cross-GPU communication
export NCCL_DEBUG=WARN  # Set to INFO for debugging
export NCCL_P2P_LEVEL=NVL  # Enable NVLink communication
export NCCL_TREE_THRESHOLD=0  # Force tree algorithm for better cross-GPU communication
export NCCL_NET_GDR_LEVEL=SYS  # System-level GPU Direct RDMA

# IMPORTANT: We need to make ALL GPUs visible to the training process
# even though DeepSpeed will only use GPUs 4-7 for the actual training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Use local rank offset to ensure training uses GPUs 4-7
# This is done by setting the local rank offset in DeepSpeed
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Run the training with proper GPU mapping
# The key is to use --include to specify which GPUs DeepSpeed should use
# Need to set PYTHONPATH to include the current directory for imports
PYTHONPATH="$(pwd):$PYTHONPATH" uv run --project training deepspeed \
    --include localhost:4,5,6,7 \
    --master_port 29501 \
    training/retrieval/train_ret.py \
    --deepspeed training/configs/rl/deepspeed_zero3.json

echo "Training complete!" 