from __future__ import annotations

from typing import List, Dict
import json
import asyncio
import os

from data.pipeline.generate_personas import generate_personas
from data.pipeline.generate_kb import generate_kb
from data.settings import OPENROUTER_GEMINI, OPENROUTER_SONNET
from data.model import get_model_response
from data.utils import load_kb_from_json
from data.schemas.kb import KnowledgeBase, Persona
from data.schemas.sft import StaticMemory
from agent.utils import load_system_prompt

# Define path directly to avoid imports from training packag
BASE_DATASET_PATH = "output/datasets/base_dataset.json"
BASE_MEMORY_PATH = "memory/base_memory"
BASE_MEMORY_JSON_PATH = "output/static_mem/base_memory.json"

QUESTION_GEN_PROMPT = """
You are {kb.items[0].persona.name_surname}. You are a {kb.items[0].persona.age} year old {kb.items[0].persona.gender} from {kb.items[0].persona.birthplace.city}, {kb.items[0].persona.birthplace.country}. You are a {kb.items[0].persona.occupation}. Your detailed backstory is: {kb.items[0].persona.detailed_backstory}.

You have the following relationships:
{kb.items[0].persona.relationships}

This is a persona related to you:
{persona}

This is a fact about the persona:
{fact}

Generate a direct question you might ask an assistant so that it reveals this fact about the persona. Keep it concise and natural. Respond with only the question. The question should be directly inquiring about the fact so the only valid answer is the fact itself. The question should NOT be an indirect question that has the possibility of being answered with the fact itself. Some example fact-question pairs:

Persona: Jane Doe, Wife of {kb.items[0].persona.name_surname}
Fact: Age: 23
Question: What is my wife's age?

Persona: Mike Doe, Son of {kb.items[0].persona.name_surname}
Fact: birthplace: Groningen, Netherlands
Question: Where was my son born?
"""

MEMORY_GEN_PROMPT = """
Below is the system prompt of an LLM agent with a self managed, Obsidian-like memory system.

<agent_prompt>
{agent_prompt}
</agent_prompt>

Below is a knowledge base of personas.

<knowledge_base>
{knowledge_base}
</knowledge_base>

The first persona in the knowledge base is the main user of the agent.

Given how you expect the agent to operate and the knowledge base, generate a static memory for the agent. This memory should contain the content of user.md and the entity files found in entities/ directory. The user.md should contain the attributes of the main user and the entity files should be for the relationships of the main user. All personas other than the main user should be in the entities/ directory.
"""

def generate_static_memory(
        knowledge_base: KnowledgeBase
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
                knowledge_base=knowledge_base
            )
        response = get_model_response(
                prompt=prompt, 
                model=OPENROUTER_GEMINI,
                schema=StaticMemory
            )
        
        return response


def generate_question_prompt(persona: Persona, fact: str, knowledge_base: KnowledgeBase) -> str:
    """
    Generate a question to elicit the given fact.

    Args:
        persona: The persona
        fact: The fact

    Returns:
        str: The question
    """
    prompt = QUESTION_GEN_PROMPT.format(persona=persona, fact=fact, kb=knowledge_base)
    response = get_model_response(prompt=prompt, model=OPENROUTER_GEMINI)
    if isinstance(response, str):
        return response.strip()
    return str(response)


def build_base_dataset(kb: KnowledgeBase, save: bool = False) -> List[Dict]:
    """
    Construct a base dataset for retrieval.

    Args:
        kb: The knowledge base

    Returns:
        List[Dict]: The base dataset
    """
    dataset: List[Dict] = []

    # Generate the static memory
    static_memory = generate_static_memory(kb)
    
    # Create a coroutine to generate content for each fact
    async def process_fact(persona, fact):
        question_task = asyncio.create_task(
            asyncio.to_thread(generate_question_prompt, persona, fact.fact_description, kb)
        )
        
        # Wait for both tasks to complete
        question = await asyncio.gather(question_task)

        return {
            "question": question,
            "answer": fact.fact_description
        }
    
    # Create tasks for all facts in all personas
    tasks = []
    new_kb = KnowledgeBase(items=kb.items[1:])
    for item in new_kb.items:
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
        with open(BASE_DATASET_PATH, "w") as f:
            json.dump(dataset, f)

        static_memory.instantiate(BASE_MEMORY_PATH)

        os.makedirs("output/static_mem", exist_ok=True)
        with open(BASE_MEMORY_JSON_PATH, "w") as f:
            json.dump(static_memory.model_dump(), f)

    return dataset


def main():
    """Generate knowledge base with personas for Groningen, Netherlands in 2025."""
    scenario = "Groningen, the Netherlands in 2025"
    print(f"Generating knowledge base for scenario: {scenario}")
    
    # Create KB with personas and save to file
    #personas = generate_personas(16, scenario, save=True)
    #kb = generate_kb(personas, save=True)
    kb = load_kb_from_json()
    
    print(f"Knowledge base generated successfully with {len(kb.items)} personas.")

    print("Building base dataset...")
    dataset = build_base_dataset(kb, save=True)
    print(f"Base dataset built successfully with {len(dataset)} items.")

if __name__ == "__main__":
    main() 