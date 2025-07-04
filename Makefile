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
	@echo "  10. generate-kb      Generate knowledge base with personas for training"
	@echo "  11. run-retrieval   Run the retrieval training"
	@echo "  12. clean-all        Remove all virtual environments and SkyRL repository"
	@echo "  13. clean-agent      Remove agent virtual environment"
	@echo "  14. clean-data       Remove data virtual environment"
	@echo "  15. clean-training   Remove training virtual environment"
	@echo "  16. clean-skyrl      Remove SkyRL repository"
	@echo "  17. run-retrieval-single  Run single agent diagnostic for retrieval training"

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
	@echo "Installing SkyRL packages..."
	@if [ ! -d "SkyRL" ]; then \
		echo "Cloning SkyRL repository..."; \
		git clone https://github.com/NovaSky-AI/SkyRL.git; \
	else \
		echo "SkyRL repository already exists, pulling latest changes..."; \
		cd SkyRL && git pull; \
	fi
	@echo "Installing skyrl-train and skyrl-gym in training environment..."
	cd training && uv add skyrl-gym
	@echo "Installing skyrl-train (trying without vllm extra first)..."
	@cd training && ( \
		uv add ../SkyRL/skyrl-train || \
		(echo "Trying with prerelease flag..." && uv add ../SkyRL/skyrl-train --prerelease=allow) || \
		(echo "Installing without vllm extra..." && uv add ../SkyRL/skyrl-train --frozen) \
	)
	@echo "Installing vllm separately if needed..."
	@cd training && (uv add 'vllm>=0.8.3' --frozen || echo "Warning: Could not install vllm, training may still work")
	@echo "Installing skyrl-gym in SkyRL/skyrl-train directory for direct execution..."
	@cd SkyRL/skyrl-train && (uv add skyrl-gym || echo "Warning: Could not install skyrl-gym in SkyRL directory")
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

# Run the retrieval training
run-retrieval:
	@echo "Running retrieval training..."
	@if [ -z "$${OBSIDIAN_ROOT}" ]; then \
		export OBSIDIAN_ROOT="$$(pwd)"; \
	fi; \
	cd SkyRL/skyrl-train && PYTHONPATH="$${OBSIDIAN_ROOT}:$$PYTHONPATH" uv run --isolated --extra vllm python "$${OBSIDIAN_ROOT}/training/retrieval/main_retrieval.py"

# Run single agent diagnostic for retrieval training
run-retrieval-single:
	@echo "Running single agent diagnostic for retrieval training..."
	@if [ -z "$${OBSIDIAN_ROOT}" ]; then \
		export OBSIDIAN_ROOT="$$(pwd)"; \
	fi; \
	cd SkyRL/skyrl-train && PYTHONPATH="$${OBSIDIAN_ROOT}:$$PYTHONPATH" uv run --isolated --extra vllm python "$${OBSIDIAN_ROOT}/training/retrieval/main_retrieval.py" --single-agent

# Generate knowledge base with personas for training
generate-kb:
	@echo "Generating knowledge base..."
	PYTHONPATH="$(PWD):$$PYTHONPATH" uv run --project data generate_kb.py

install-verifiers: 
	uv sync && \
	uv add 'verifiers[all]' && \
	uv pip install flash-attn==2.7.4.post1 --no-build-isolation

run-vf-llm:
	CUDA_VISIBLE_DEVICES=$(VF_INFERENCE_GPUS) NCCL_P2P_LEVEL=NVL uv run vf-vllm --model $(VF_MODEL)

run-vf-ret:
	CUDA_VISIBLE_DEVICES=$(VF_TRAINING_GPUS) NCCL_P2P_LEVEL=NVL uv run accelerate launch --config-file training/configs/rl/zero3.yaml --gpu_ids 0,1,2,3 run_vf_ret.py

# Clean all virtual environments
clean-all: clean-agent clean-data clean-training clean-skyrl

# Clean agent virtual environment
clean-agent:
	cd agent && rm -rf .venv

# Clean data virtual environment  
clean-data:
	cd data && rm -rf .venv

# Clean training virtual environment
clean-training:
	cd training && rm -rf .venv

# Clean SkyRL repository
clean-skyrl:
	rm -rf SkyRL