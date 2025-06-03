from typing import Optional, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona, Fact
from data.model import SFTModel

from agent.agent import Agent
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


def generate_conversation_with_retries(
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
        setup_func(**kwargs)
    
    success = conversation_generator_func(**kwargs)
    if success:
        return True
    
    for _ in range(max_retries):
        if setup_func:
            setup_func(**kwargs)
        success = conversation_generator_func(**kwargs)
        if success:
            return True
    
    return False


def generate_conversation_for_persona(
        persona_model: BaseSFTModel,
        agent: Agent,
        num_turns: int,
        facts_to_check: list[Fact]
    ) -> bool:
    """
    Generate a conversation between a persona model and an agent.
    
    Args:
        persona_model: The persona model
        agent: The agent
        num_turns: Number of conversation turns
        facts_to_check: Facts to verify in memory after conversation
    
    Returns:
        bool: True if conversation was successful and facts were recorded
    """
    # Create the memory if it doesn't exist
    create_memory_if_not_exists()
    
    persona_message = persona_model.chat()
    
    # Generate the conversation
    for turn in tqdm(range(num_turns), desc="Conversation turns", unit="turn", leave=False):
        agent_response = agent.chat(persona_message)
        persona_message = persona_model.chat(agent_response.reply)
    
    # Check if the facts are present in the memory
    folder_dump_str = dump_folder(MEMORY_PATH)
    reward = get_reward(folder_dump_str=folder_dump_str, facts_to_check=facts_to_check)
    if reward < 0.99:
        delete_memory()
        return False
    
    # Save the conversation and delete the memory
    agent.save_conversation()
    delete_memory()
    return True


def generate_sft_for_kb(
        kb: KnowledgeBase,
        conversation_func: Callable,
        num_turns: int = 4,
        max_retries: int = 3,
        setup_func: Optional[Callable] = None,
        **kwargs
    ) -> None:
    """
    Generate SFT dataset for a knowledge base.
    
    Args:
        kb: The knowledge base
        conversation_func: Function that generates conversation for persona and fact
        num_turns: Number of conversation turns
        max_retries: Maximum number of retries
        setup_func: Optional setup function to call before each attempt
        **kwargs: Additional arguments for the conversation function
    """
    for kb_item in tqdm(kb.items, desc="Processing personas", unit="persona"):
        persona = kb_item.persona
        facts = kb_item.facts
        
        for fact in tqdm(facts, desc=f"Generating conversations for {persona.name_surname}", unit="fact", leave=False):
            success = generate_conversation_with_retries(
                conversation_func,
                max_retries=max_retries,
                setup_func=setup_func,
                persona=persona,
                fact=fact,
                num_turns=num_turns,
                **kwargs
            )
            if not success:
                continue 