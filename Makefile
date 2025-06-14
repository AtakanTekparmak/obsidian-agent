# Set default target
.DEFAULT_GOAL := help

# Variables

## Python and uv
PYTHON := python3

# Verifiers variables
VF_MODEL := Qwen/Qwen3-8B
VF_TENSOR_PARALLEL_SIZE := 4
VF_MAX_BATCH_SIZE := 128
VF_INFERENCE_GPUS := 0,1,2,3
VF_TRAINING_GPUS := 4,5,6,7
VF_ALL_GPUS := 0,1,2,3,4,5,6,7
VF_NUM_PROCESSES := 4

# Help target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  1. check-uv          Check if uv is installed and install if needed"
	@echo "  2. install-all       Install all three environments (agent, data, training)"
	@echo "  3. install-agent     Install only agent environment"
	@echo "  4. install-data      Install only data environment" 
	@echo "  5. install-training  Install only training environment"
	@echo "  6. copy-env          Copy the .env.example file to .env if it doesn't exist"
	@echo "  7. run-agent         Run the agent"
	@echo "  8. generate-data     Generate data using the pipeline"
	@echo "  9. build-dataset     Build the HF dataset and upload it to the Hub"
	@echo "  10. vf-inference     Start verifiers inference server"
	@echo "  11. vf-training      Start verifiers training (uses DeepSpeed directly)"
	@echo "  12. vf-training-all-gpus  Start verifiers training with all GPUs visible"
	@echo "  13. vf-generate-kb   Generate knowledge base with personas for training"
	@echo "  14. clean-all        Remove all virtual environments"
	@echo "  15. clean-agent      Remove agent virtual environment"
	@echo "  16. clean-data       Remove data virtual environment"
	@echo "  17. clean-training   Remove training virtual environment"

# Check if uv is installed and install if needed
check-uv:
	@echo "Checking if uv is installed..."
	@if ! command -v uv > /dev/null; then \
		echo "uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Please restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) to use uv"; \
	else \
		echo "uv is already installed"; \
		uv --version; \
	fi

# Install all three environments
install-all: check-uv install-agent install-data install-training

# Install agent environment
install-agent: check-uv
	@echo "Setting up agent environment..."
	cd agent && uv sync
	@echo "Agent environment setup complete!"

# Install data environment  
install-data: check-uv
	@echo "Setting up data environment..."
	cd data && uv sync
	@echo "Data environment setup complete!"

# Install training environment
install-training: check-uv
	@echo "Setting up training environment..."
	cd training && uv sync
	@echo "Installing flash-attn..."
	cd training && uv add verifiers[all] && uv pip install flash-attn --no-build-isolation
	@echo "Training environment setup complete!"

# Copy the .env.example file to .env if it doesn't exist
copy-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from .env.example"; \
	else \
		echo ".env file already exists"; \
	fi

# Generate data using the pipeline
generate-data:
	PYTHONPATH="$(PWD):$$PYTHONPATH" uv run --project data generate_data.py

# Run the agent
run-agent:
	PYTHONPATH="$(PWD):$$PYTHONPATH" uv run --project agent run_agent.py

# Build the HF dataset and upload it to the Hub
build-dataset:
	PYTHONPATH="$(PWD):$$PYTHONPATH" uv run --project data build_hf_dataset.py --data_dir output/conversations

# Start verifiers inference server
vf-inference:
	@echo "Starting verifiers inference server..."
	PYTHONPATH="$(PWD):$$PYTHONPATH" CUDA_VISIBLE_DEVICES=$(VF_INFERENCE_GPUS) uv run --project training vf-vllm --model $(VF_MODEL) --tensor-parallel-size $(VF_TENSOR_PARALLEL_SIZE) --max-batch-size $(VF_MAX_BATCH_SIZE)

# Start verifiers training
vf-training:
	@echo "Starting verifiers training..."
	@echo "Make sure vf-inference is running in another terminal first!"
	./training/scripts/run_verifiers_training.sh

# Alternative: Run training with all GPUs visible for NCCL communication
vf-training-all-gpus:
	@echo "Starting verifiers training with all GPUs visible..."
	@echo "Make sure vf-inference is running in another terminal first!"
	@echo "This method allows NCCL communication between inference (GPUs 0-3) and training (GPUs 4-7)"
	PYTHONPATH="$(PWD):$$PYTHONPATH" CUDA_VISIBLE_DEVICES=$(VF_ALL_GPUS) CUDA_DEVICE_ORDER=PCI_BUS_ID uv run --project training accelerate launch \
		--config-file verifiers/configs/zero3.yaml \
		--num-processes $(VF_NUM_PROCESSES) \
		--gpu_ids 4,5,6,7 \
		--main_process_port 29501 \
		training/retrieval/train_retrieval.py

# Generate knowledge base with personas for training
vf-generate-kb:
	@echo "Generating knowledge base..."
	PYTHONPATH="$(PWD):$$PYTHONPATH" uv run --project data generate_kb.py

# Clean all virtual environments
clean-all: clean-agent clean-data clean-training

# Clean agent virtual environment
clean-agent:
	cd agent && rm -rf .venv

# Clean data virtual environment  
clean-data:
	cd data && rm -rf .venv

# Clean training virtual environment
clean-training:
	cd training && rm -rf .venv