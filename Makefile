# Set default target
.DEFAULT_GOAL := help

# Variables

## Python
PYTHON := python3
PIP := pip3
VENV_NAME := venv

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
	@echo "  6. clean             Remove the virtual environment and its contents"

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

# Clean the virtual environment
clean:
	rm -rf $(VENV_NAME)