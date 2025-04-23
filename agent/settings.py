import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-2.5-pro-exp-03-25"

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_STRONG_MODEL = "google/gemini-2.5-pro-preview-03-25"

# Memory
MEMORY_PATH = "memory_dir"
FILE_SIZE_LIMIT = 1024 * 1024 # 1MB
DIR_SIZE_LIMIT = 1024 * 1024 * 10 # 10MB
MEMORY_SIZE_LIMIT = 1024 * 1024 * 100 # 100MB

# Path settings
SYSTEM_PROMPT_PATH = "agent/system_prompt.txt"