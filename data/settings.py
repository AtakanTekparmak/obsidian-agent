import os
from dotenv import load_dotenv
from dria import Model

# Load environment variables
load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# Dria
STRUCTURED_OUTPUT_MODELS = [Model.GPT4O]
STRONG_MODELS = [
    Model.DEEPSEEK_CHAT_OR,
    Model.LLAMA_3_1_405B_OR,
    Model.ANTHROPIC_SONNET_3_5_OR,
    Model.GPT4O
]

# Constants
NUM_PERSONAS_PER_SCENARIO = 8

# File paths
OUTPUT_PATH = "output"
PERSONAS_PATH = "personas"
RELATIONSHIPS_PATH = "relationships"

# Seed data
SCENARIO = "Amsterdam, Netherlands in 2024"