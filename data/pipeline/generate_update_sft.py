from typing import Union, Optional
from random import choice
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona
from data.schemas.sft import StaticMemory
from data.model import get_model_response
from data.settings import OPENROUTER_SONNET

from agent.agent import Agent
from agent.utils import delete_memory, load_system_prompt
from agent.schemas import ChatMessage, Role

PROMPT = """
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

def generate_static_memory(
        persona: Persona, 
        fact: str
    ) -> StaticMemory:
        """
        Generate a static memory for the agent.
        """
        prompt = PROMPT.format(
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