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
	@echo "  2. run               Run the main.py script (should be run third)"
	@echo "  3. test              Run the tests"
	@echo "  4. clean             Remove the virtual environment and its contents"

# Install dependencies and set up the environment
install:
	$(PYTHON) -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && \
	$(PIP) install -r requirements.txt 

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