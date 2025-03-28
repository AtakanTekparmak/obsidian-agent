from dria import Model

# Models
STRUCTURED_OUTPUT_MODELS = [Model.GPT4O]
STRONG_MODELS = [
    Model.DEEPSEEK_CHAT_OR,
    Model.LLAMA_3_1_405B_OR,
    Model.ANTHROPIC_SONNET_3_5_OR,
    Model.GPT4O
]

# Constants
NUM_PERSONAS_PER_SCENARIO = 1

# File paths
OUTPUT_PATH = "output"
PERSONAS_PATH = "personas"

# Seed data
SCENARIOS = [
    "A modern town in 2024, Amsterdam, Netherlands",
    "A neighborhood in 2025, Hamburg, Germany"
]