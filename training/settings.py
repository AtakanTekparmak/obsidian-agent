from dotenv import load_dotenv

load_dotenv()

# File paths
JUDGE_PROMPT_PATH = "training/reward/judge_prompt.txt"
VERIFIERS_DATASET_PATH = "output/datasets/verifiers_dataset.json"

# Models
GEMINI_PRO = "gemini-2.5-pro-exp-03-25"
GEMINI_FLASH = "gemini-2.5-flash-preview-04-17"
GPT_4O = "gpt-4o-2024-11-20"
O4_MINI = "o4-mini-2025-04-16"
GPT_O3 = "o3-2025-04-16"