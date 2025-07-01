import os
import json
import uuid

from training.reward.utils import construct_folder_judge_prompt, construct_reply_judge_prompt
from training.reward.model import get_model_response
from training.reward.schemas import JudgeResponse, ReplyJudgeResponse
from training.settings import GPT_O3, JUDGE_CONVERSATION_SAVE_PATH

from data.schemas.kb import Fact

from pydantic import BaseModel

class JudgeConversation(BaseModel):
    agent_reply: str
    ground_truth: str
    judge_response: ReplyJudgeResponse

def save_judge_conversation(judge_conversation: JudgeConversation):
    """
    Save the judge conversation to the file.

    Args:
        judge_conversation: The judge conversation to save.
    """
    try:
        # Make sure the directory exists
        os.makedirs(JUDGE_CONVERSATION_SAVE_PATH, exist_ok=True)
        judge_uuid = str(uuid.uuid4())
        try:
            with open(os.path.join(JUDGE_CONVERSATION_SAVE_PATH, f"judge_conversation_{judge_uuid}.json"), "w") as f:
                json.dump(judge_conversation.model_dump(mode='json'), f, indent=2)
        except Exception as e:
            print(f"Error saving judge conversation to {os.path.join(JUDGE_CONVERSATION_SAVE_PATH, f'judge_conversation_{judge_uuid}.json')}: {e}")
            raise
    except Exception as e:
        print(f"Error saving judge conversation: {e}")
        raise


def get_folder_reward(
        folder_dump_str: str,
        facts_to_check: list[Fact]
    ) -> float:
    """
    Get the LLM-as-a-judge reward for the given folder dump and facts to check.

    Args:
        folder_dump_str: The folder dump as a string.
        facts_to_check: The facts to check.

    Returns:
        The reward as a float. The reward is the ratio of the number of facts present in the folder dump to the total number of facts.
    """
    judge_prompt = construct_folder_judge_prompt(folder_dump_str, facts_to_check)
    judge_response = get_model_response(
        schema=JudgeResponse,
        prompt=judge_prompt,
        model=GPT_O3
    )
    return judge_response.ratio_of_facts_present

def get_reward(
        agent_reply: str,
        ground_truth: str,
        debug: bool = False
    ) -> float:
    """
    Get the reward for the given agent reply and ground truth.
    
    Returns:
        float: 1.0 if ground truth is present in reply, 0.0 otherwise
    """
    judge_prompt = construct_reply_judge_prompt(agent_reply, ground_truth)
    judge_response = get_model_response(
        schema=ReplyJudgeResponse,
        prompt=judge_prompt,
        model=GPT_O3
    )

    if debug:
        judge_conversation = JudgeConversation(
            agent_reply=agent_reply,
            ground_truth=ground_truth,
            judge_response=judge_response
        )
        save_judge_conversation(judge_conversation)

    # Convert boolean to float (1.0 for True, 0.0 for False)
    return 1.0 if judge_response.ground_truth_in_reply else 0.0