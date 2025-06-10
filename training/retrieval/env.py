from __future__ import annotations

import uuid
from typing import Any, Dict, List, Tuple, Union

from datasets import Dataset
from verifiers.envs.environment import Environment
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric

from agent.agent import Agent
from agent.utils import delete_memory
from data.schemas.sft import StaticMemory

class RetrievalEnv(Environment):
    """Environment that runs the Obsidian agent to answer retrieval questions."""

    def __init__(self, dataset: Dataset | List[Dict], rubric: Rubric, parser: Parser | None = None, **kwargs):
        if parser is None:
            parser = Parser()
        super().__init__(dataset=Dataset.from_list(dataset) if not isinstance(dataset, Dataset) else dataset,
                         parser=parser, rubric=rubric, **kwargs)
        self.agent_cls = Agent

    def rollout(
        self,
        client,
        model,
        prompt: Union[str, List[Dict[str, Any]]],
        answer: str,
        sampling_args: Dict[str, Any] | None = None,
        static_memory: StaticMemory | None = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        memory_path = f"memory_{uuid.uuid4().hex}"
        if static_memory is not None:
            static_memory.instantiate(memory_path)
        agent = self.agent_cls(memory_path=memory_path)
        agent.chat(prompt if isinstance(prompt, str) else prompt[-1]["content"])
        messages = [msg.model_dump() for msg in agent.messages]
        delete_memory(memory_path)
        return messages, {}

    def generate(
        self,
        inputs: Dict[str, List[Any]] | Dataset,
        client=None,
        model: str | None = None,
        sampling_args: Dict[str, Any] | None = None,
        max_concurrent: int | None = None,
        score_rollouts: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(inputs, Dataset):
            results = {col: list(inputs[col]) for col in inputs.column_names}
        else:
            results = {k: list(v) for k, v in inputs.items()}
        prompts = results["prompt"]
        answers = results["answer"]
        static_memories = results.get("static_memory", [None] * len(prompts))
        completions = []
        states = []
        for prompt, answer, memory in zip(prompts, answers, static_memories):
            comp, state = self.rollout(client, model, prompt, answer, static_memory=memory)
            completions.append(comp)
            states.append(state)
        results["completion"] = completions
        results["state"] = states
        tasks = results.get("task", ["default"] * len(prompts))
        if score_rollouts:
            reward_dict = self.rubric.score_rollouts(
                prompts=prompts,
                completions=completions,
                answers=answers,
                states=states,
                tasks=tasks,
            )
            results.update(reward_dict)
        return results
