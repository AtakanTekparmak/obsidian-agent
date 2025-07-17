from training.settings import FOLDER_JUDGE_PROMPT_PATH, REPLY_JUDGE_PROMPT_PATH
from data.schemas.kb import Fact

def load_folder_judge_prompt() -> str:
    """
    Load the judge prompt from the file.

    Returns:
        The judge prompt as a string.
    """
    try:
        with open(FOLDER_JUDGE_PROMPT_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Judge prompt file not found at {FOLDER_JUDGE_PROMPT_PATH}")
    
def load_reply_judge_prompt() -> str:
    """
    Load the reply judge prompt from the file.
    """
    try:
        with open(REPLY_JUDGE_PROMPT_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Reply judge prompt file not found at {REPLY_JUDGE_PROMPT_PATH}")
    
def construct_folder_judge_prompt(
        folder_dump_str: str,
        facts_to_check: list[Fact]
    ) -> str:
    """
    Construct the judge prompt by substituting the placeholders with the actual values.

    Args:
        folder_dump_str: The folder dump as a string.
        facts_to_check: The facts to check.

    Returns:
        The constructed judge prompt.
    """
    judge_prompt = load_folder_judge_prompt()
    return judge_prompt.replace("{{folder_dump_str}}", folder_dump_str).replace("{{facts_to_check}}", "\n".join([fact.model_dump_json() for fact in facts_to_check]))

def construct_reply_judge_prompt(
        reply: str,
        ground_truth: str
    ) -> str:
    """
    Construct the reply judge prompt by substituting the placeholders with the actual values.
    """
    judge_prompt = load_reply_judge_prompt()
    return judge_prompt.replace("{{reply}}", reply).replace("{{ground_truth}}", ground_truth)
