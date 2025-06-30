import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Get the absolute path to the training directory
TRAINING_DIR = Path(__file__).parent.absolute()
OBSIDIAN_ROOT = TRAINING_DIR.parent

# File paths - now absolute
FOLDER_JUDGE_PROMPT_PATH = str(OBSIDIAN_ROOT / "training" / "reward" / "prompts" / "judge_prompt.txt")
REPLY_JUDGE_PROMPT_PATH = str(OBSIDIAN_ROOT / "training" / "reward" / "prompts" / "reply_judge_prompt.txt")
VERIFIERS_DATASET_PATH = str(OBSIDIAN_ROOT / "output" / "datasets" / "verifiers_dataset.json")

# Models
GEMINI_PRO = "gemini-2.5-pro-exp-03-25"
GEMINI_FLASH = "gemini-2.5-flash-preview-04-17"
GPT_4O = "gpt-4o-2024-11-20"
O4_MINI = "o4-mini-2025-04-16"
GPT_O3 = "o3-2025-04-16"