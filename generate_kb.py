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

from training.retrieval.format_dataset import main as format_dataset

# Define path directly to avoid imports from training package
SKRL_DATASET_PATH = "output/datasets/skrl_dataset.json"

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

Given how you expect the agent to operate, the persona & the fact, generate a static memory for the agent. This memory should contain the content of user.md and the entity files found in entities/ directory. The user.md should contain the attributes of the persona and the entity files should be for the relationships of the persona.
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


def build_skyrl_dataset(kb: KnowledgeBase, save: bool = False) -> List[Dict]:
    """
    Construct a SkyRL dataset for retrieval.

    Args:
        kb: The knowledge base

    Returns:
        List[Dict]: The SkyRL dataset
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
             "data_source": "obsidian-retrieval",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "env_class": "obsidian-retrieval",
            "reward_spec": {
                "method": "rule",
                "ground_truth": fact.fact_description
            },
            "extra_info": {
                "static_memory": static_memory.model_dump_json()
            }
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
        with open(SKRL_DATASET_PATH, "w") as f:
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

    print("Building SkyRL dataset...")
    dataset = build_skyrl_dataset(kb, save=True)
    print(f"SkyRL dataset built successfully with {len(dataset)} items.")

    print("Formatting dataset...")
    format_dataset()
    print("Dataset formatted successfully.")

if __name__ == "__main__":
    main() 