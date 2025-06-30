#!/bin/bash

# SkyRL Training Script for Obsidian Retrieval Environment
# Based on the SkyRL multiply example adapted for our retrieval task

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Dataset paths
DATA_DIR="output/datasets/skyrl_formatted"
TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="$DATA_DIR/validation.parquet"

# Check if formatted data exists
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "Formatted datasets not found. Running dataset generation and formatting..."
    make generate-kb
    echo "Datasets prepared successfully"
fi

# Model configuration
MODEL_PATH="Qwen/Qwen3-4B"

# Training configuration
EXPERIMENT_NAME="obsidian-retrieval-qwen3-4b"
OUTPUT_DIR="./output/training/${EXPERIMENT_NAME}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting SkyRL training for Obsidian Retrieval Environment"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# Ensure environment is registered
echo "Registering obsidian-retrieval environment..."
PYTHONPATH="$(pwd):$PYTHONPATH" uv run --project training python -c "import training; print('Environment registration complete')"

# Run training with SkyRL
# Use array to build command arguments - only override specific values
CMD_ARGS=(
    "data.train_data=['$TRAIN_DATA']"
    "data.val_data=['$VAL_DATA']"
    "trainer.policy.model.path=$MODEL_PATH"    
    "trainer.run_name=$EXPERIMENT_NAME"
    "trainer.output_dir=$OUTPUT_DIR"
)

PYTHONPATH="$(pwd):$PYTHONPATH" uv run --project training training/retrieval/main_retrieval.py "${CMD_ARGS[@]}"

echo "Training completed! Check outputs in: $OUTPUT_DIR" 