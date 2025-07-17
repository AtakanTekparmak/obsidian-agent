from data.schemas.kb import KnowledgeBase, Persona, Fact
from data.settings import MAX_CONCURRENT_PERSONAS, MAX_CONCURRENT_FACTS
from agent.async_agent import AsyncAgent

from .base import (
    BaseSFTModel,
    generate_conversation_for_persona,
    generate_sft_for_kb,
    default_fact_validation,
)

PERSONA_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is the fact you will be providing to the LLM assistant:
{fact}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. Your goal is to have a natural conversation with the LLM assistant, and in one of the message you will provide the fact you are given. You should not be too direct/forthcoming about the fact you're trying to provide. Given the fact you are going to provide, you should start the conversation and steer it so you can provide the fact in a natural way. You have {num_turns} of allowed conversation, and you should choose in which message you will provide the fact.

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. The conversation HAS to be in English, no matter which persona you are.
"""


class PersonaModel(BaseSFTModel):
    """
    Utility class for an LLM assuming the role of a persona.
    It is used to generate a conversation for a persona and a fact.
    """

    def __init__(self, persona: Persona, fact: str, num_turns: int):
        self.fact = fact
        super().__init__(persona, num_turns)

    def _get_system_prompt(self, persona: Persona, num_turns: int) -> str:
        return PERSONA_PROMPT.format(
            persona=persona, fact=self.fact, num_turns=num_turns
        )


async def generate_convo_for_persona_and_fact(
    persona: Persona,
    fact: Fact,
    num_turns: int,
    validation_func=default_fact_validation,
    memory_path: str = None,
    save_folder: str = None,
) -> bool:
    """
    Generate a conversation for a persona and a fact.

    Args:
        persona: The persona
        fact: The fact
        num_turns: The number of turns
        validation_func: Function to validate conversation results
        memory_path: The memory path for the agent
        save_folder: Folder name to save conversations to

    Returns:
        bool: True if the conversation was generated successfully, False otherwise
    """
    persona_model = PersonaModel(
        persona=persona, fact=fact.fact_description, num_turns=num_turns
    )
    agent = AsyncAgent(memory_path=memory_path)

    return await generate_conversation_for_persona(
        persona_model=persona_model,
        agent=agent,
        num_turns=num_turns,
        facts_to_check=[fact],
        validation_func=validation_func,
        save_folder=save_folder,
    )


async def generate_introduce_sft(
    kb: KnowledgeBase,
    num_turns: int = 4,
    max_retries: int = 3,
    validation_func=default_fact_validation,
    save_folder: str = "introduce",
    max_concurrent_personas: int = MAX_CONCURRENT_PERSONAS,
    max_concurrent_facts: int = MAX_CONCURRENT_FACTS,
) -> None:
    """
    Generate a SFT dataset by the agent interacting
    with the user in a multiturn conversations

    Args:
        kb: The knowledge base
        num_turns: The number of turns
        max_retries: The number of retries
        validation_func: Function to validate conversation results
        save_folder: Folder name to save conversations to
        max_concurrent_personas: Maximum number of personas to process concurrently
        max_concurrent_facts: Maximum number of facts per persona to process concurrently

    Returns:
        None
    """
    await generate_sft_for_kb(
        kb=kb,
        conversation_func=generate_convo_for_persona_and_fact,
        num_turns=num_turns,
        max_retries=max_retries,
        validation_func=validation_func,
        save_folder=save_folder,
        task_name="introduce",
        max_concurrent_personas=max_concurrent_personas,
        max_concurrent_facts=max_concurrent_facts,
    )
