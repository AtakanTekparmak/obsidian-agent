# Set default target
.DEFAULT_GOAL := help

# Variables

## Python
PYTHON := python3
PIP := pip3
VENV_NAME := venv

# Verifiers variables
VF_MODEL := Qwen/Qwen3-8B
VF_TENSOR_PARALLEL_SIZE := 4
VF_MAX_BATCH_SIZE := 128
VF_INFERENCE_GPUS := 0,1,2,3
VF_TRAINING_GPUS := 4,5,6,7
VF_NUM_PROCESSES := 4

# Help target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  1. install           Install dependencies and set up the environment (should be run first)"
	@echo "  2. copy-env          Copy the .env.example file to .env if it doesn't exist (should be run second)"
	@echo "  3. run-agent         Run the agent"
	@echo "  4. generate-data     Generate data using the pipeline"
	@echo "  5. build-dataset     Build the HF dataset and upload it to the Hub"
	@echo "  6. vf-install        Install verifiers with uv and project dependencies"
	@echo "  7. vf-inference      Start verifiers inference server"
	@echo "  8. vf-training       Start verifiers training"
	@echo "  9. clean             Remove the virtual environment and its contents"

# Install dependencies and set up the environment
install:
	$(PYTHON) -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && \
	$(PIP) install -r requirements.txt 

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
	. $(VENV_NAME)/bin/activate && \
	$(PYTHON) generate_data.py

# Run the agent
run-agent:
	. $(VENV_NAME)/bin/activate && \
	$(PYTHON) run_agent.py

# Build the HF dataset and upload it to the Hub
build-dataset:
	. $(VENV_NAME)/bin/activate && \
	$(PYTHON) build_hf_dataset.py --data_dir output/conversations

# Install verifiers with uv and project dependencies
vf-install:
	@echo "Checking if uv is installed..."
	@if ! command -v uv > /dev/null; then \
		echo "uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv is already installed"; \
	fi
	@echo "Setting up uv environment for verifiers..."
	cd verifiers && uv sync --extra all
	@echo "Installing flash-attn..."
	cd verifiers && uv pip install flash-attn --no-build-isolation
	@echo "Installing project requirements..."
	cd verifiers && cat ../requirements.txt | grep -v "^#" | grep -v "^$$" | xargs -I {} uv add "{}"
	@echo "Verifiers environment setup complete!"

# Start verifiers inference server
vf-inference:
	@echo "Starting verifiers inference server..."
	cd verifiers && CUDA_VISIBLE_DEVICES=$(VF_INFERENCE_GPUS) uv run vf-vllm --model $(VF_MODEL) --tensor-parallel-size $(VF_TENSOR_PARALLEL_SIZE) --max-batch-size $(VF_MAX_BATCH_SIZE)

# Start verifiers training
vf-training:
	@echo "Starting verifiers training..."
	CUDA_VISIBLE_DEVICES=$(VF_TRAINING_GPUS) ./verifiers/.venv/bin/accelerate launch --config-file verifiers/configs/zero3.yaml --num-processes $(VF_NUM_PROCESSES) training/retrieval/train_retrieval.py

# Clean the virtual environment
clean:
	rm -rf $(VENV_NAME)