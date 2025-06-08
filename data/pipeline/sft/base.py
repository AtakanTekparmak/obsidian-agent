from typing import Optional, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona, Fact
from data.model import SFTModel
from data.settings import MAX_CONCURRENT_PERSONAS, MAX_CONCURRENT_FACTS

import asyncio
import uuid
from agent.async_agent import AsyncAgent
from agent.utils import delete_memory, create_memory_if_not_exists
from agent.schemas import ChatMessage, Role
from agent.settings import MEMORY_PATH

from training.reward import dump_folder, get_reward


class BaseSFTModel(SFTModel, ABC):
    """
    Base class for SFT models that assume the role of a persona.
    """
    def __init__(self, persona: Persona, num_turns: int):
        super().__init__(num_turns)
        self.persona = persona
        self.messages: list[ChatMessage] = [
            ChatMessage(
                role=Role.SYSTEM, 
                content=self._get_system_prompt(persona, num_turns)
            )
        ]
    
    @abstractmethod
    def _get_system_prompt(self, persona: Persona, num_turns: int) -> str:
        """Generate the system prompt for this model type."""
        pass


def default_fact_validation(facts_to_check: list[Fact], memory_path: str = MEMORY_PATH) -> bool:
    """
    Default validation function that checks if facts are present in memory.
    
    Args:
        facts_to_check: List of facts to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    folder_dump_str = dump_folder(memory_path)
    reward = get_reward(folder_dump_str=folder_dump_str, facts_to_check=facts_to_check)
    return reward >= 0.99


async def _maybe_await(func: Callable, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


async def generate_conversation_with_retries(
        conversation_generator_func: Callable,
        max_retries: int = 3,
        setup_func: Optional[Callable] = None,
        **kwargs
    ) -> bool:
    """
    Generate a conversation with retries on failure.
    
    Args:
        conversation_generator_func: Function that generates a conversation
        max_retries: Maximum number of retries
        setup_func: Optional setup function to call before each retry
        **kwargs: Arguments to pass to the conversation generator function
    
    Returns:
        bool: True if conversation was generated successfully, False otherwise
    """
    if setup_func:
        await _maybe_await(setup_func, **kwargs)

    success = await _maybe_await(conversation_generator_func, **kwargs)
    if success:
        return True

    for _ in range(max_retries):
        if setup_func:
            await _maybe_await(setup_func, **kwargs)
        success = await _maybe_await(conversation_generator_func, **kwargs)
        if success:
            return True
    
    return False


async def generate_conversation_for_persona(
        persona_model: BaseSFTModel,
        agent: AsyncAgent,
        num_turns: int,
        facts_to_check: list[Fact],
        validation_func: Callable[[list[Fact], str], bool] = default_fact_validation,
        save_folder: str = None
    ) -> bool:
    """
    Generate a conversation between a persona model and an agent.
    
    Args:
        persona_model: The persona model
        agent: The agent
        num_turns: Number of conversation turns
        facts_to_check: Facts to verify after conversation
        validation_func: Function to validate the conversation results
        save_folder: Folder name to save conversations to
    
    Returns:
        bool: True if conversation was successful and validation passed
    """
    # Create the memory if it doesn't exist
    create_memory_if_not_exists(agent.memory_path)

    persona_message = await persona_model.achat()
    
    # Generate the conversation
    for turn in range(num_turns):
        agent_response = await agent.chat(persona_message)
        # Only continue the conversation if agent provided a reply
        if agent_response.reply:
            persona_message = await persona_model.achat(agent_response.reply)
        else:
            # If agent doesn't reply, break the conversation loop
            return False
    
    # Validate the conversation results
    if not validation_func(facts_to_check, agent.memory_path):
        delete_memory(agent.memory_path)
        return False

    # Save the conversation and delete the memory
    await agent.save_conversation(save_folder=save_folder)
    delete_memory(agent.memory_path)
    return True


async def generate_sft_for_kb(
        kb: KnowledgeBase,
        conversation_func: Callable,
        num_turns: int = 4,
        max_retries: int = 3,
        setup_func: Optional[Callable] = None,
        validation_func: Callable[[list[Fact], str], bool] = default_fact_validation,
        save_folder: str = None,
        task_name: str = "",
        max_concurrent_personas: int = MAX_CONCURRENT_PERSONAS,
        max_concurrent_facts: int = MAX_CONCURRENT_FACTS,
        **kwargs
    ) -> None:
    """
    Generate SFT dataset for a knowledge base with concurrent processing.
    
    Args:
        kb: The knowledge base
        conversation_func: Function that generates conversation for persona and fact
        num_turns: Number of conversation turns
        max_retries: Maximum number of retries
        setup_func: Optional setup function to call before each attempt
        validation_func: Function to validate conversation results
        save_folder: Folder name to save conversations to
        task_name: Name of the task for progress bar description
        max_concurrent_personas: Maximum number of personas to process concurrently
        max_concurrent_facts: Maximum number of facts per persona to process concurrently
        **kwargs: Additional arguments for the conversation function
    """
    # Create semaphores to control concurrency
    persona_semaphore = asyncio.Semaphore(max_concurrent_personas)
    fact_semaphore = asyncio.Semaphore(max_concurrent_facts)
    
    async def process_fact_for_persona(persona: Persona, fact: Fact) -> bool:
        """Process a single fact for a persona with fact-level concurrency control."""
        async with fact_semaphore:
            memory_path = f"{persona.name_surname.replace(' ', '_')}_{uuid.uuid4().hex}"
            success = await generate_conversation_with_retries(
                conversation_func,
                max_retries=max_retries,
                setup_func=setup_func,
                persona=persona,
                fact=fact,
                num_turns=num_turns,
                validation_func=validation_func,
                memory_path=memory_path,
                save_folder=save_folder,
                **kwargs
            )
            return success
    
    async def process_persona(kb_item) -> int:
        """Process all facts for a single persona concurrently."""
        async with persona_semaphore:
            persona = kb_item.persona
            facts = kb_item.facts
            
            # Process all facts for this persona concurrently
            fact_tasks = [
                process_fact_for_persona(persona, fact) 
                for fact in facts
            ]
            
            if fact_tasks:
                # Use tqdm for progress tracking at the fact level
                fact_results = []
                with tqdm(total=len(fact_tasks), 
                         desc=f"Processing facts for {persona.name_surname}", 
                         unit="fact", 
                         leave=False) as pbar:
                    
                    # Process facts with progress updates
                    for coro in asyncio.as_completed(fact_tasks):
                        result = await coro
                        fact_results.append(result)
                        pbar.update(1)
                
                successful_facts = sum(1 for result in fact_results if result)
                return successful_facts
            return 0
    
    # Process all personas concurrently
    task_desc = f"Processing personas for {task_name}" if task_name else "Processing personas"
    
    persona_tasks = [process_persona(kb_item) for kb_item in kb.items]
    
    if persona_tasks:
        total_successful_facts = 0
        with tqdm(total=len(persona_tasks), desc=task_desc, unit="persona") as pbar:
            for coro in asyncio.as_completed(persona_tasks):
                successful_facts = await coro
                total_successful_facts += successful_facts
                pbar.update(1)
        
        print(f"Completed {task_name}: {total_successful_facts} successful conversations generated")
