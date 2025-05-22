from typing import Union, Optional

from data.schemas.kb import KnowledgeBase, PersonaWithStories

from agent.agent import Agent
from agent.utils import delete_memory
from agent.schemas import ChatMessage, Role
from agent.model import get_model_response

PERSONA_PROMPT = """
You are {persona.name_surname}. You are a {persona.age} year old {persona.gender} from {persona.birthplace.city}, {persona.birthplace.country}. You are a {persona.occupation}. Your detailed backstory is: {persona.detailed_backstory}. 

You have the following relationships:
{persona.relationships}

You have the following stories:
{stories}

You will be conversing with an LLM assistant that has a self managed, Obsidian-like memory system. Your goal is to have a natural conversation with the LLM assistant, while also providing the LLM assistant with information about yourself (without being too obvious about it). You should not be too direct/forthcoming about the information you're trying to provide, and you should only provide information if you think it can be relevant to the conversation. You necessarily don't need to provide all the information about yourself, but have a genuine every day conversation with the LLM assistant while providing the assistant with some information that can be relevant to the conversation.

You should start the conversation now. Don't be verbose, don't forget the LLM assistant is an AI assistant. Don't say more than 2 sentences at a time, that number is absolute. 
"""

class PersonaModel():
    """
    Utility class for an LLM assuming the role of a persona.
    """
    def __init__(self, persona_with_stories: PersonaWithStories):
        self.persona_with_stories = persona_with_stories
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=PERSONA_PROMPT.format(persona=self.persona_with_stories.persona, stories=self.persona_with_stories.stories))
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
        num_turns: int = 10
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
    for kb_item in kb.items:
        persona_with_stories = kb_item.persona_with_stories
        persona_model = PersonaModel(persona_with_stories)
        agent = Agent()
        persona_message = persona_model.chat()

        for _ in range(num_turns):
            agent_message = agent.chat(persona_message).agent_response_2
            persona_message = persona_model.chat(agent_message)

        agent.save_conversation()
        delete_memory()

    