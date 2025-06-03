from typing import Union, Optional
from random import choice

from data.schemas.kb import KnowledgeBase, Persona, Fact
from data.schemas.sft import StaticMemory, FactUpdate
from data.model import get_model_response
from data.settings import OPENROUTER_SONNET

from agent.agent import Agent
from agent.utils import load_system_prompt

from .base import (
    BaseSFTModel, 
    generate_conversation_for_persona, 
    generate_sft_for_kb
)

# Prompts
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

Given how you expect the agent to operate, the persona & the fact, generate a static memory for the agent. This memory should contain the content of a guideline file (always found in guideline.md), the path to the user file (in a structure similar to user/user_name.md) and the content of the user file. Make sure to only have the given fact in the memory, nothing else related to the user.
"""

UPDATE_GEN_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is a fact about you:
{fact}

You will generate an update to the fact. This update to the original fact should be within reason and faithful to the persona. The update should match the style and syntax of the original fact. IF the fact you're trying to update is not sensible to update, like the birthplace of a person or their parental relationship(s), or any other fact that according to the persona (and to you) is not sensible to update, you should set the fact_update_possible to False. Otherwise, set it to True. In the end, you should return the initial fact, the updated fact and the fact_update_possible field.
"""

SFT_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is a fact about you:
{fact_update.initial_fact}

This is the update to the fact:
{fact_update.updated_fact}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. Your goal is to have a natural conversation with the LLM assistant, and in one of the messages you will provide the update to the fact you are given. You should not be too direct/forthcoming about the update you're trying to provide. Given the update you are going to provide, you should start the conversation and steer it so you can provide the update in a natural way. You have {num_turns} of allowed conversation, and you should choose in which message you will provide the update. 

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. The conversation HAS to be in English, no matter which persona you are. 
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

def generate_fact_update(
        persona: Persona, 
        fact: str
    ) -> FactUpdate:
        """
        Generate a fact update for the agent.

        Args:
            persona: The persona
            fact: The fact

        Returns:
            FactUpdate: The fact update
        """
        prompt = UPDATE_GEN_PROMPT.format(
                persona=persona, 
                fact=fact
            )
        response = get_model_response(
                prompt=prompt, 
                model=OPENROUTER_SONNET,
                schema=FactUpdate
            )
        
        return response

class UpdateModel(BaseSFTModel):
    """
    Utility class for an LLM assuming the role of a persona
    that is going to provide an update to an existing fact.
    """
    def __init__(self, persona: Persona, fact_update: FactUpdate, num_turns: int):
        self.fact_update = fact_update
        super().__init__(persona, num_turns)
    
    def _get_system_prompt(self, persona: Persona, num_turns: int) -> str:
        return SFT_PROMPT.format(
            persona=persona, 
            fact_update=self.fact_update, 
            num_turns=num_turns
        )

def generate_convo_for_persona_and_update(
        persona: Persona,
        fact_update: FactUpdate,
        num_turns: int
    ) -> bool:
        """
        Generate a conversation for a persona and a fact update.

        Args:
            persona: The persona
            fact_update: The fact update
            num_turns: The number of turns

        Returns:
            bool: True if the conversation was generated successfully, False otherwise
        """
        update_model = UpdateModel(
            persona=persona,    
            fact_update=fact_update, 
            num_turns=num_turns
        )
        agent = Agent()

        updated_fact = Fact(fact_description=fact_update.updated_fact)
        
        return generate_conversation_for_persona(
            persona_model=update_model,
            agent=agent,
            num_turns=num_turns,
            facts_to_check=[updated_fact]
        )

# Global variables to store state for retry attempts
_fact_update_cache = {}
_static_memory_cache = {}

def _setup_static_memory(persona: Persona, fact: Fact, **kwargs):
    """Setup function that creates static memory for each retry attempt."""
    cache_key = (persona.name_surname, fact.fact_description)
    
    if cache_key not in _static_memory_cache:
        static_memory = generate_static_memory(
            persona=persona, 
            fact=fact.fact_description
        )
        _static_memory_cache[cache_key] = static_memory
    
    _static_memory_cache[cache_key].instantiate()

def _generate_update_conversation(
        persona: Persona,
        fact: Fact,
        num_turns: int
    ) -> bool:
    """
    Helper function to generate update conversation.
    """
    cache_key = (persona.name_surname, fact.fact_description)
    
    # Generate or retrieve cached fact update
    if cache_key not in _fact_update_cache:
        fact_update = generate_fact_update(
            persona=persona, 
            fact=fact.fact_description
        )
        if not fact_update.fact_update_possible:
            return False
        _fact_update_cache[cache_key] = fact_update
    
    fact_update = _fact_update_cache[cache_key]
    
    return generate_convo_for_persona_and_update(
        persona=persona, 
        fact_update=fact_update, 
        num_turns=num_turns
    )

def generate_update_sft(
        kb: KnowledgeBase,
        num_turns: int = 4,
        max_retries: int = 3
    ) -> None:
        """
        Generate a SFT dataset by the agent interacting
        with the user in a multiturn conversations  

        Args:
            kb: The knowledge base
            num_turns: The number of turns
            max_retries: The number of retries

        Returns:
            None
        """
        # Clear caches
        global _fact_update_cache, _static_memory_cache
        _fact_update_cache.clear()
        _static_memory_cache.clear()
        
        generate_sft_for_kb(
            kb=kb,
            conversation_func=_generate_update_conversation,
            setup_func=_setup_static_memory,
            num_turns=num_turns,
            max_retries=max_retries
        )