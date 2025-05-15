from training.settings import JUDGE_PROMPT_PATH
from data.schemas.kb import Fact

def load_judge_prompt() -> str:
    """
    Load the judge prompt from the file.

    Returns:
        The judge prompt as a string.
    """
    try:
        with open(JUDGE_PROMPT_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Judge prompt file not found at {JUDGE_PROMPT_PATH}")
    
def construct_judge_prompt(
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
    judge_prompt = load_judge_prompt()
    return judge_prompt.replace("{{folder_dump_str}}", folder_dump_str).replace("{{facts_to_check}}", "\n".join([fact.model_dump_json() for fact in facts_to_check]))
    