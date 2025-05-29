from typing import Union, Optional
from random import choice
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona
from data.schemas.sft import StaticMemory
from data.model import get_model_response, SFTModel
from data.settings import OPENROUTER_SONNET

from agent.agent import Agent
from agent.utils import delete_memory, load_system_prompt
from agent.schemas import ChatMessage, Role

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

SFT_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is a fact about you:
{fact}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. Your goal is to have a natural conversation with the LLM assistant, and in one of the messages you will provide an update to the fact you are given. You should not be too direct/forthcoming about the update you're trying to provide. Given the update you are going to provide, you should start the conversation and steer it so you can provide the update in a natural way. You have {num_turns} of allowed conversation, and you should choose in which message you will provide the update. An example fact-update pair would be:
- Fact: lives in Noorderplantsoenbuurt
- Update: Moved to Indischebuurt

You should decide on the update before starting the conversation.

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. 
"""

def generate_static_memory(
        persona: Persona, 
        fact: str
    ) -> StaticMemory:
        """
        Generate a static memory for the agent.
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