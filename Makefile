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
	@cd training && (uv add vllm || echo "Warning: Could not install vllm, training may still work")
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
	chmod +x run_retrieval.sh
	./run_retrieval.sh

# Generate knowledge base with personas for training
generate-kb:
	@echo "Generating knowledge base..."
	PYTHONPATH="$(PWD):$$PYTHONPATH" uv run --project data generate_kb.py

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