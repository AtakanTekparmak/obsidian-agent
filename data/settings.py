import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH = "gemini-2.5-flash-preview-04-17"
GEMINI_PRO = "gemini-2.5-pro-exp-03-25"

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_STRONG_MODEL = "google/gemini-2.5-pro-preview-03-25"

# OpenAI
GPT_4O = "gpt-4o-2024-11-20"
O4_MINI = "o4-mini-2025-04-16"

# File paths
OUTPUT_PATH = "output"
PERSONAS_PATH = "personas"
STORIES_PATH = "stories"
KB_PATH = "kb"
CONVO_PATH = "convos"
SFT_PATH = "sft"