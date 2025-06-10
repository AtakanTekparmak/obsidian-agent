from __future__ import annotations

from typing import List, Dict

from data.pipeline.generate_personas import generate_personas
from data.pipeline.generate_kb import generate_kb
from data.pipeline.sft.generate_update_sft import generate_static_memory
from data.settings import OPENROUTER_SONNET
from data.model import get_model_response
from data.schemas.kb import KnowledgeBase, Persona

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


def build_verifiers_dataset(kb: KnowledgeBase) -> List[Dict]:
    """
    Construct a verifiers dataset for retrieval.

    Args:
        kb: The knowledge base

    Returns:
        List[Dict]: The verifiers dataset
    """
    dataset: List[Dict] = []
    for item in kb.items:
        persona = item.persona
        for fact in item.facts:
            static_memory = generate_static_memory(persona, fact.fact_description)
            question = generate_question_prompt(persona, fact.fact_description)
            dataset.append(
                {
                    "prompt": question,
                    "answer": fact.fact_description,
                    "task": "retrieval",
                    "static_memory": static_memory,
                    "persona": persona.name_surname,
                    "fact": fact.fact_description,
                }
            )
    return dataset


def create_kb_with_personas(num_personas: int, scenario: str, save: bool = False) -> KnowledgeBase:
    """
    Utility to generate personas and KB in one go.

    Args:
        num_personas: The number of personas to generate
        scenario: The scenario to generate the personas for
    """
    personas = generate_personas(num_personas, scenario, save=save)
    kb = generate_kb(personas, save=save)
    return kb
