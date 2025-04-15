import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-2.5-pro-exp-03-25"

# Memory
MEMORY_PATH = "memory"