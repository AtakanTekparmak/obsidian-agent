from __future__ import annotations

from typing import List, Dict
import json
import asyncio

from data.pipeline.generate_personas import generate_personas
from data.pipeline.generate_kb import generate_kb
from data.pipeline.sft.generate_update_sft import generate_static_memory
from data.settings import OPENROUTER_SONNET
from data.model import get_model_response
from data.schemas.kb import KnowledgeBase, Persona

# Define path directly to avoid importing from training.settings which triggers training.__init__.py
VERIFIERS_DATASET_PATH = "output/datasets/verifiers_dataset.json"

QUESTION_GEN_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}.

You have the following relationships:
{persona.relationships}

This is a fact about you:
{fact}

Generate a direct question you might ask an assistant so that it reveals this fact about you. Keep it concise and natural. Respond with only the question.
"""


def generate_question_prompt(persona: Persona, fact: str) -> str:
    """
    Generate a question to elicit the given fact.

    Args:
        persona: The persona
        fact: The fact

    Returns:
        str: The question
    """
    prompt = QUESTION_GEN_PROMPT.format(persona=persona, fact=fact)
    response = get_model_response(prompt=prompt, model=OPENROUTER_SONNET)
    if isinstance(response, str):
        return response.strip()
    return str(response)


def build_verifiers_dataset(kb: KnowledgeBase, save: bool = False) -> List[Dict]:
    """
    Construct a verifiers dataset for retrieval.

    Args:
        kb: The knowledge base

    Returns:
        List[Dict]: The verifiers dataset
    """
    dataset: List[Dict] = []

    # Create a coroutine to generate content for each fact
    async def process_fact(persona, fact):
        # Run these operations concurrently
        static_memory_task = asyncio.create_task(
            asyncio.to_thread(generate_static_memory, persona, fact.fact_description)
        )
        question_task = asyncio.create_task(
            asyncio.to_thread(generate_question_prompt, persona, fact.fact_description)
        )

        # Wait for both tasks to complete
        static_memory, question = await asyncio.gather(
            static_memory_task, question_task
        )

        return {
            "question": question,
            "answer": fact.fact_description,
            "task": "retrieval",
            "static_memory": static_memory,
            "persona": persona.name_surname,
        }

    # Create tasks for all facts in all personas
    tasks = []
    for item in kb.items:
        persona = item.persona
        for fact in item.facts:
            tasks.append(process_fact(persona, fact))

    # Run all tasks concurrently and collect results
    results = asyncio.run(asyncio.gather(*tasks))
    dataset.extend(results)
    if save:
        with open(VERIFIERS_DATASET_PATH, "w") as f:
            json.dump(dataset, f)
    return dataset


def load_verifiers_dataset(path: str = VERIFIERS_DATASET_PATH) -> List[Dict]:
    """
    Load a verifiers dataset from a JSON file.

    Args:
        path: The path to the verifiers dataset

    Returns:
        List[Dict]: The verifiers dataset
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return []


def create_kb_with_personas(
    num_personas: int, scenario: str, save: bool = False
) -> KnowledgeBase:
    """
    Utility to generate personas and KB in one go.

    Args:
        num_personas: The number of personas to generate
        scenario: The scenario to generate the personas for
    """
    personas = generate_personas(num_personas, scenario, save=save)
    kb = generate_kb(personas, save=save)
    return kb
