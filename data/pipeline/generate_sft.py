from typing import Union, Optional
from random import choice
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona

from agent.agent import Agent
from agent.utils import delete_memory
from agent.schemas import ChatMessage, Role
from agent.model import get_model_response

PERSONA_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is the fact you will be providing to the LLM assistant:
{fact}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. Your goal is to have a natural conversation with the LLM assistant, and in one of the message you will provide the fact you are given. You should not be too direct/forthcoming about the fact you're trying to provide. Given the fact you are going to provide, you should start the conversation and steer it so you can provide the fact in a natural way. You have {num_turns} of allowed conversation, and you should choose in which message you will provide the fact.

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. 
"""

class PersonaModel():
    """
    Utility class for an LLM assuming the role of a persona.
    """
    def __init__(self, persona: Persona, fact: str, num_turns: int, prompt: Optional[str] = None):
        self.persona = persona
        self.fact = fact
        self.num_turns = num_turns
        if not prompt:
            self.messages: list[ChatMessage] = [
                ChatMessage(role=Role.SYSTEM, content=PERSONA_PROMPT.format(persona=self.persona, fact=self.fact, num_turns=self.num_turns))
            ]
        else:
            self.messages: list[ChatMessage] = [
                ChatMessage(role=Role.SYSTEM, content=prompt)
            ]

    def _add_message(self, message: Union[ChatMessage, dict]):
        """ Add a message to the conversation history. """
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
        else:
            raise ValueError("Invalid message type")

    def chat(self, message: Optional[str] = None) -> str:
        """ Chat with the LLM assistant. """
        if message:
            self._add_message(ChatMessage(role=Role.USER, content=message))

        response = get_model_response(messages=self.messages)
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response))

        return response

def generate_sft(
        kb: KnowledgeBase,
        num_turns: int = 4
    ) -> None:
    """
    Generate a SFT dataset by the agent interacting
    with the user in a multiturn conversations  

    Args:
        convos: The multiturn conversations
        save: Whether to save the SFT dataset

    Returns:
        None
    """
    for kb_item in tqdm(kb.items, desc="Processing personas", unit="persona"):

        persona = kb_item.persona
        facts = kb_item.facts

        for fact in tqdm(facts, desc=f"Generating conversations for {persona.name_surname}", unit="fact", leave=False):
            error = False
            persona_model = PersonaModel(persona, fact, num_turns)
            agent = Agent()
            persona_message = persona_model.chat()

            for turn in tqdm(range(num_turns), desc="Conversation turns", unit="turn", leave=False):
                agent_response = agent.chat(persona_message)
                agent_message = agent_response.agent_response_2
                if agent_response.error:
                    error = True
                    break
                persona_message = persona_model.chat(agent_message)

            if error:
                print(f"Error for {persona.name_surname} with fact {fact}, skipping...")
                continue

            agent.save_conversation()
            delete_memory()

    