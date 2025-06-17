from typing import List

from verifiers import RewardFunc
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

from training.reward import get_reward
from data.schemas.kb import Fact

class MemoryRubric(Rubric):
    """Simple rubric checking that the answer appears in the reply."""

    def __init__(self, funcs: List[RewardFunc] = [], weights: List[float] = [], parser: XMLParser | None = None, env_parser: XMLParser | None = None):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        if not isinstance(self.parser, XMLParser):
            self.parser = XMLParser(["think", "python", "reply"], answer_field="reply")
        self.env_parser = env_parser if env_parser is not None else XMLParser(["result"])
        self.add_reward_func(self.answer_in_reply_reward_func)

    def answer_in_reply_reward_func(self, completion, answer, **kwargs) -> float:
        response = str(self.parser.parse_answer(completion)).lower()
        fact = Fact(fact_description=answer)
        return get_reward(folder_dump_str=response, facts=[fact])