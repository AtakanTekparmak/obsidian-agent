import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_STRONG_MODEL = "google/gemini-2.5-pro-preview-03-25"

# Memory
MEMORY_PATH = "memory_dir"
FILE_SIZE_LIMIT = 1024 * 1024 # 1MB
DIR_SIZE_LIMIT = 1024 * 1024 * 10 # 10MB
MEMORY_SIZE_LIMIT = 1024 * 1024 * 100 # 100MB

# Engine
SANDBOX_TIMEOUT = 20

# Path settings
SYSTEM_PROMPT_PATH = "agent/system_prompt.txt"
SAVE_CONVERSATION_PATH = "output/conversations/"