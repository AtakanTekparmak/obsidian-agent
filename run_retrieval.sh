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

# Run training with SkyRL
# Use array to build command arguments
CMD_ARGS=(
    "environment.env_class=obsidian-retrieval"
    "data.train_data=['$TRAIN_DATA']"
    "data.val_data=['$VAL_DATA']"
    "trainer.policy.model.path=$MODEL_PATH"
    "trainer.policy.model.trust_remote_code=true"
    "trainer.placement.colocate_all=true"
    "trainer.placement.policy_num_gpus_per_node=4"
    "generator.num_inference_engines=2"
    "generator.inference_engine_tensor_parallel_size=2"
    "trainer.algorithm.advantage_estimator=grpo"
    "trainer.algorithm.use_kl_loss=false"
    "trainer.epochs=2"
    "generator.n_samples_per_prompt=4"
    "generator.sampling_params.temperature=0.7"
    "generator.sampling_params.top_p=0.9"
    "generator.sampling_params.max_generate_length=2048"
    "trainer.policy.optimizer_config.lr=1.0e-6"
    "trainer.train_batch_size=64"
    "trainer.policy_mini_batch_size=32"
    "trainer.micro_forward_batch_size_per_gpu=4"
    "trainer.micro_train_batch_size_per_gpu=2"
    "trainer.eval_batch_size=32"
    "trainer.eval_before_train=true"
    "trainer.eval_interval=1"
    "trainer.max_prompt_length=8192"
    "generator.max_input_length=8192"
    "trainer.logger=wandb"
    "trainer.wandb_project=obsidian-retrieval-skyrl"
    "trainer.wandb_run_name=$EXPERIMENT_NAME"
    "trainer.output_dir=$OUTPUT_DIR"
    "trainer.save_strategy=epoch"
    "trainer.save_total_limit=4"
    "generator.async_engine=false"
    "generator.batched=true"
    "generator.max_turns=8"
)

PYTHONPATH="$(pwd):$PYTHONPATH" uv run --project training -m skyrl_train.entrypoints.main_base "${CMD_ARGS[@]}"

echo "Training completed! Check outputs in: $OUTPUT_DIR" 