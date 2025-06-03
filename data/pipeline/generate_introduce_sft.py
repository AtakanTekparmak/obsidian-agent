from typing import Optional
from tqdm import tqdm

from data.schemas.kb import KnowledgeBase, Persona, Fact
from data.model import SFTModel

from agent.agent import Agent
from agent.utils import delete_memory, create_memory_if_not_exists
from agent.schemas import ChatMessage, Role
from agent.settings import MEMORY_PATH

from training.reward import dump_folder, get_reward

PERSONA_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

This is the fact you will be providing to the LLM assistant:
{fact}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. Your goal is to have a natural conversation with the LLM assistant, and in one of the message you will provide the fact you are given. You should not be too direct/forthcoming about the fact you're trying to provide. Given the fact you are going to provide, you should start the conversation and steer it so you can provide the fact in a natural way. You have {num_turns} of allowed conversation, and you should choose in which message you will provide the fact.

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. The conversation HAS to be in English, no matter which persona you are.
"""

class PersonaModel(SFTModel):
    """
    Utility class for an LLM assuming the role of a persona.
    It is used to generate a conversation for a persona and a fact.
    """
    def __init__(self, persona: Persona, fact: str, num_turns: int):
        super().__init__(num_turns)
        self.messages: list[ChatMessage] = [
            ChatMessage(
                role=Role.SYSTEM, 
                content=PERSONA_PROMPT.format(
                    persona=persona, 
                    fact=fact, 
                    num_turns=num_turns
                )
            )
        ]

def generate_convo_for_persona_and_fact(
        persona: Persona,
        fact: Fact,
        num_turns: int
    ) -> bool:
        """
        Generate a conversation for a persona and a fact.

        Args:
            persona: The persona
            fact: The fact
            num_turns: The number of turns

        Returns:
            bool: True if the conversation was generated successfully, False otherwise
        """
        # Instantiate the persona model and the agent
        persona_model = PersonaModel(
            persona=persona, 
            fact=fact.fact_description, 
            num_turns=num_turns
        )
        agent = Agent()

        # Create the memory if it doesn't exist
        create_memory_if_not_exists()

        persona_message = persona_model.chat()

        # Generate the conversation
        for turn in tqdm(range(num_turns), desc="Conversation turns", unit="turn", leave=False):
            agent_response = agent.chat(persona_message)
            agent_message = agent_response.agent_response_2

            if agent_response.error:
                print(f"Error: {agent_response.error}")
                return False
            
            persona_message = persona_model.chat(agent_message)

        # Check if the fact is present in the memory
        folder_dump_str = dump_folder(MEMORY_PATH)
        reward = get_reward(folder_dump_str=folder_dump_str, facts_to_check=[fact])
        if reward < 0.99:
            delete_memory()
            return False
        
        # Save the conversation and delete the memory
        agent.save_conversation()
        delete_memory()
        return True


def generate_introduce_sft(
        kb: KnowledgeBase,
        num_turns: int = 4,
        max_retries: int = 3
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
            convo_success = generate_convo_for_persona_and_fact(
                persona=persona, 
                fact=fact, 
                num_turns=num_turns
            )
            if not convo_success:
                for _ in range(max_retries):
                    convo_success = generate_convo_for_persona_and_fact(
                        persona=persona, 
                        fact=fact, 
                        num_turns=num_turns
                    )
                    if convo_success:
                        break
                if not convo_success:
                    continue
    