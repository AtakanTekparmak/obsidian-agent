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
OPENROUTER_SONNET = "anthropic/claude-sonnet-4"
OPENROUTER_OPUS   = "anthropic/claude-opus-4"
OPENROUTER_GEMINI = "google/gemini-2.5-pro"

# OpenAI
GPT_4O = "gpt-4o-2024-11-20"
O4_MINI = "o4-mini-2025-04-16"
GPT_4_5 = "gpt-4.5-preview-2025-02-27"

# File paths
OUTPUT_PATH = "output"
PERSONAS_PATH = "personas"
STORIES_PATH = "stories"
KB_PATH = "kb"
CONVO_PATH = "convos"
SFT_PATH = "sft"

# SFT Pipeline concurrency settings
MAX_CONCURRENT_PERSONAS = 4
MAX_CONCURRENT_FACTS = 8
