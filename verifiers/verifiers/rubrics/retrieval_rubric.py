from __future__ import annotations

from typing import List

from .rubric import Rubric
from verifiers import RewardFunc

from training.reward import get_reward
from data.schemas.kb import Fact
from agent.schemas import AgentResponse


def retrieval_reward(completion: List[dict] | str, answer: str, **kwargs) -> float:
    """
    Reward function for retrieval.

    Args:
        completion: The completion
        answer: The answer
    """
    if isinstance(completion, list):
        last_content = completion[-1]["content"] if completion else ""
    else:
        last_content = completion

    try:
        response = AgentResponse.model_validate_json(last_content)
        text = response.reply or ""
    except Exception:
        text = last_content

    fact = Fact(fact_description=answer)
    return get_reward(folder_dump_str=text, facts_to_check=[fact])


def get_retrieval_rubric() -> Rubric:
    return Rubric(funcs=[retrieval_reward], weights=[1.0]) 