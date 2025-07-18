from typing import List
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona, Fact
from data.schemas.sft import StaticMemory
from data.settings import MAX_CONCURRENT_PERSONAS, MAX_CONCURRENT_FACTS

from agent.async_agent import AsyncAgent
from agent.utils import delete_memory, create_memory_if_not_exists

from training.reward import get_folder_reward

from .base import BaseSFTModel, generate_sft_for_kb
from .generate_update_sft import generate_static_memory

RETRIEVE_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is a fact about you that the LLM assistant should know:
{fact}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. The assistant already has information about you stored in its memory. Your goal is to have a natural conversation with the LLM assistant and ask a direct question where the assistant would need to retrieve the fact from memory and reply with the exact fact. 

You should lead the conversation naturally and then ask a specific question about the fact. The assistant should be able to retrieve and provide the exact fact from its memory. You have {num_turns} turns of conversation.

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. The conversation HAS to be in English, no matter which persona you are.
"""


class RetrieveModel(BaseSFTModel):
    """
    Utility class for an LLM assuming the role of a persona
    that will ask the agent to retrieve a fact from memory.
    """

    def __init__(self, persona: Persona, fact: str, num_turns: int):
        self.fact = fact
        super().__init__(persona, num_turns)

    def _get_system_prompt(self, persona: Persona, num_turns: int) -> str:
        return RETRIEVE_PROMPT.format(
            persona=persona, fact=self.fact, num_turns=num_turns
        )


def retrieval_validation(facts_to_check: List[Fact], agent_replies: List[str]) -> bool:
    """
    Custom validation function for retrieval that checks if the agent
    successfully retrieved and replied with the facts.

    Args:
        facts_to_check: List of facts to validate
        agent_replies: List of agent replies from the conversation

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Combine all agent replies as the "folder dump" for validation
    combined_replies = "\n\n".join(agent_replies)
    
    # Use the get_folder_reward function to check if facts are present in replies
    reward = get_folder_reward(folder_dump_str=combined_replies, facts_to_check=facts_to_check)
    return reward >= 0.99


async def generate_retrieve_conversation_for_persona(
    persona_model: RetrieveModel,
    agent: AsyncAgent,
    num_turns: int,
    facts_to_check: List[Fact],
    save_folder: str = None,
) -> bool:
    """
    Generate a retrieval conversation between a persona model and an agent.

    Args:
        persona_model: The persona model
        agent: The agent
        num_turns: Number of conversation turns
        facts_to_check: Facts to verify in agent replies
        save_folder: Folder name to save conversations to

    Returns:
        bool: True if conversation was successful and validation passed
    """
    # Create the memory if it doesn't exist
    create_memory_if_not_exists(agent.memory_path)

    agent_replies = []

    persona_message = await persona_model.achat()

    # Generate the conversation and collect agent replies
    for turn in range(num_turns):
        agent_response = await agent.chat(persona_message)

        # Collect agent replies for validation
        if agent_response.reply:
            agent_replies.append(agent_response.reply)
            persona_message = await persona_model.achat(agent_response.reply)
        else:
            # If agent doesn't reply, break the conversation loop
            return False

    # Validate using custom retrieval validation
    if not retrieval_validation(facts_to_check, agent_replies):
        delete_memory(agent.memory_path)
        return False

    # Save the conversation and delete the memory
    await agent.save_conversation(save_folder=save_folder)
    delete_memory(agent.memory_path)
    return True


async def generate_convo_for_persona_and_retrieve(
    persona: Persona,
    fact: Fact,
    num_turns: int,
    validation_func=None,  # Added parameter to match expected signature
    memory_path: str = None,
    save_folder: str = None,
) -> bool:
    """
    Generate a retrieval conversation for a persona and a fact.

    Args:
        persona: The persona
        fact: The fact to retrieve
        num_turns: The number of turns
        validation_func: Unused, kept for interface compatibility
        memory_path: The memory path for the agent
        save_folder: Folder name to save conversations to

    Returns:
        bool: True if the conversation was generated successfully, False otherwise
    """
    retrieve_model = RetrieveModel(
        persona=persona, fact=fact.fact_description, num_turns=num_turns
    )
    agent = AsyncAgent(memory_path=memory_path)

    return await generate_retrieve_conversation_for_persona(
        persona_model=retrieve_model,
        agent=agent,
        num_turns=num_turns,
        facts_to_check=[fact],
        save_folder=save_folder,
    )


class RetrieveSFTCache:
    """Clean cache management for retrieve SFT generation."""

    def __init__(self):
        self.static_memories = {}

    def clear(self):
        """Clear all cached data."""
        self.static_memories.clear()

    def get_or_create_static_memory(self, persona: Persona, fact: Fact) -> StaticMemory:
        """Get cached static memory or create a new one."""
        cache_key = (persona.name_surname, fact.fact_description)

        if cache_key not in self.static_memories:
            static_memory = generate_static_memory(
                persona=persona, fact=fact.fact_description
            )
            self.static_memories[cache_key] = static_memory

        return self.static_memories[cache_key]


def _setup_static_memory_with_cache(
    cache: RetrieveSFTCache, persona: Persona, fact: Fact, memory_path: str, **kwargs
):
    """Setup function that creates static memory for each retry attempt."""
    static_memory = cache.get_or_create_static_memory(persona, fact)
    static_memory.instantiate(memory_path)


async def _generate_retrieve_conversation_with_cache(
    cache: RetrieveSFTCache,
    persona: Persona,
    fact: Fact,
    num_turns: int,
    memory_path: str,
    save_folder: str = None,
) -> bool:
    """Helper function to generate retrieve conversation with cache."""
    return await generate_convo_for_persona_and_retrieve(
        persona=persona,
        fact=fact,
        num_turns=num_turns,
        memory_path=memory_path,
        save_folder=save_folder,
    )


async def generate_retrieve_sft(
    kb: KnowledgeBase,
    num_turns: int = 4,
    max_retries: int = 4,
    save_folder: str = "retrieve",
    max_concurrent_personas: int = MAX_CONCURRENT_PERSONAS,
    max_concurrent_facts: int = MAX_CONCURRENT_FACTS,
) -> None:
    """
    Generate a SFT dataset for retrieval by having the agent
    retrieve facts from memory in response to direct questions.

    Args:
        kb: The knowledge base
        num_turns: The number of turns
        max_retries: The number of retries
        save_folder: Folder name to save conversations to
        max_concurrent_personas: Maximum number of personas to process concurrently
        max_concurrent_facts: Maximum number of facts per persona to process concurrently

    Returns:
        None
    """
    cache = RetrieveSFTCache()

    # Create wrapper functions that include the cache
    async def conversation_func_with_cache(
        persona,
        fact,
        num_turns,
        validation_func=None,
        memory_path=None,
        save_folder=None,
    ):
        return await _generate_retrieve_conversation_with_cache(
            cache, persona, fact, num_turns, memory_path, save_folder
        )

    async def setup_func_with_cache(persona, fact, memory_path=None, **kwargs):
        return _setup_static_memory_with_cache(
            cache, persona, fact, memory_path, **kwargs
        )

    # Use a dummy validation function since retrieve has custom validation
    def dummy_validation(facts_to_check):
        return True  # Actual validation happens inside conversation generation

    await generate_sft_for_kb(
        kb=kb,
        conversation_func=conversation_func_with_cache,
        setup_func=setup_func_with_cache,
        num_turns=num_turns,
        max_retries=max_retries,
        validation_func=dummy_validation,
        save_folder=save_folder,
        task_name="retrieve",
        max_concurrent_personas=max_concurrent_personas,
        max_concurrent_facts=max_concurrent_facts,
    )
