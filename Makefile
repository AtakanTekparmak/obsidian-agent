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
	@echo "  3. run               Run the main.py script (should be run third)"
	@echo "  4. test              Run the tests"
	@echo "  5. clean             Remove the virtual environment and its contents"

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

# Run the main.py script
run:
	. $(VENV_NAME)/bin/activate && \
	$(PYTHON) main.py

# Run the tests
test:
	. $(VENV_NAME)/bin/activate && \
	$(PYTHON) tests/run_tests.py

# Clean the virtual environment
clean:
	rm -rf $(VENV_NAME)