from __future__ import annotations

from typing import List, Dict
import json
import asyncio
import os

from data.pipeline.generate_personas import generate_personas
from data.pipeline.generate_kb import generate_kb
from data.settings import OPENROUTER_SONNET
from data.model import get_model_response
from data.utils import load_kb_from_json
from data.schemas.kb import KnowledgeBase, Persona
from data.schemas.sft import StaticMemory
from agent.utils import load_system_prompt

# Define path directly to avoid imports from training package
VERIFIERS_DATASET_PATH = "output/datasets/verifiers_dataset.json"

QUESTION_GEN_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}.

You have the following relationships:
{persona.relationships}

This is a fact about you:
{fact}

Generate a direct question you might ask an assistant so that it reveals this fact about you. Keep it concise and natural. Respond with only the question. The question should be directly inquiring about the fact so the only valid answer is the fact itself. The question should NOT be an indirect question that has the possibility of being answered with the fact itself. Some example fact-question pairs:

Fact: Age: 23
Question: What is my age?

Fact: birthplace: Groningen, Netherlands
Question: Where was I born?
"""

MEMORY_GEN_PROMPT = """
Below is the system prompt of an LLM agent with a self managed, Obsidian-like memory system.

<agent_prompt>
{agent_prompt}
</agent_prompt>

Below is a persona.

<persona>
{persona}
</persona>

Below is a fact about the persona.

<fact>
{fact}
</fact>

Given how you expect the agent to operate, the persona & the fact, generate a static memory for the agent. This memory should contain the content of a guideline file (always found in guideline.md), the path to the user file (in a structure similar to user/user_name.md) and the content of the user file. Make sure to only have the given fact in the memory, nothing else related to the user. The guideline should explicitly state that the main & active user is the persona.
"""


def generate_static_memory(
        persona: Persona,
        fact: str
    ) -> StaticMemory:
        """
        Generate a static memory for the agent.

        Args:
            persona: The persona
            fact: The fact

        Returns:
            StaticMemory: The static memory
        """
        prompt = MEMORY_GEN_PROMPT.format(
                agent_prompt=load_system_prompt(), 
                persona=persona, 
                fact=fact
            )
        response = get_model_response(
                prompt=prompt, 
                model=OPENROUTER_SONNET,
                schema=StaticMemory
            )
        
        return response


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
        static_memory, question = await asyncio.gather(static_memory_task, question_task)
        
        return {
            "prompt": question,
            "answer": fact.fact_description,
            "task": "retrieval",
            "static_memory": static_memory.model_dump_json(),
            "persona": persona.name_surname,
            "fact": fact.fact_description,
        }
    
    # Create tasks for all facts in all personas
    tasks = []
    for item in kb.items:
        persona = item.persona
        for fact in item.facts:
            tasks.append(process_fact(persona, fact))
    
    # Run all tasks concurrently and collect results
    async def gather_all():
        return await asyncio.gather(*tasks)
    
    results = asyncio.run(gather_all())
    dataset.extend(results)
    if save:
        # Make the output/datasets folder if it doesn't exist
        os.makedirs("output/datasets", exist_ok=True)
        with open(VERIFIERS_DATASET_PATH, "w") as f:
            json.dump(dataset, f)
    return dataset


def main():
    """Generate knowledge base with personas for Groningen, Netherlands in 2025."""
    scenario = "Groningen, the Netherlands in 2025"
    print(f"Generating knowledge base for scenario: {scenario}")
    
    # Create KB with personas and save to file
    # personas = generate_personas(8, scenario, save=True)
    # kb = generate_kb(personas, save=True)
    kb = load_kb_from_json()
    
    print(f"Knowledge base generated successfully with {len(kb.items)} personas.")

    print("Building verifiers dataset...")
    dataset = build_verifiers_dataset(kb, save=True)
    print(f"Verifiers dataset built successfully with {len(dataset)} items.")

if __name__ == "__main__":
    main() 