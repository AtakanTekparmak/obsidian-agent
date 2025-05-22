from data.schemas.convo import MultiTurnConvos

from agent.agent import Agent
from agent.utils import delete_memory

def generate_sft(
        convos: MultiTurnConvos
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
    for convo in convos.convos:
        agent = Agent()
        for chat in convo.user_chats:
            _ = agent.chat(chat.user_message)

        agent.save_conversation()
        delete_memory()

    