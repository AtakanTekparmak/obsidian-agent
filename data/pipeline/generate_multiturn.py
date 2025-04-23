import os
from typing import List

from data.model import get_model_response
from data.settings import OUTPUT_PATH, CONVO_PATH, GEMINI_PRO
from data.utils import save_pydantic_to_json
from data.schemas.convo import MultiTurnConvos
from data.schemas.kb import KnowledgeBase

def generate_multiturn_convos(
        kb: KnowledgeBase,
        save: bool = True
    ) -> MultiTurnConvos:
    """
    Generate a list of user turns for a multiturn conversation
    using the knowledge base.

    Args:
        kb: KnowledgeBase object
        save: Whether to save the multiturn conversations to a file

    Returns:
        MultiTurnConvos object
    """
    
    prompt = f"Looking at the knowledge base with personas, their relationships, facts and stories about them, for each persona, generate a list of user turns that would be used for a multi-turn user-LLM agent/assistant conversation that organically, naturally and seamlessly delivers all the facts about the persona. In the data you generate, include just the user turns/what the user would say to the LLM assistant in a natural conversation, the facts that you have chosen to deliver to the assistant and shall be checked if the assistant knows after the conversation has ended. Make sure to generate only what the user would say and make sure that the user is not deliberately trying to give the factual information all the time, but more like having a natural conversation with the LLM assistant. Pay great attention to the overall coherence of the conversation. Don't just talk about something different every user turn but assume that a memory organising agent has recorded the important information from the last turn and you are continuing the conversation without jumping to a completely different topic too abruptly. Make sure all the base_fact in the user_turns are present in the facts_to_check, and nothing more. Don't add facts to the facts_to_check_list that you haven't provided in the conversation. In one chat turn, only present one base_fact. Don't assume anything about the assistant agent's capabilities, except that it has a self managed, Obsidian-like memory system. Do not forget to assume a persona as the user of the conversation. Don't put direct timestamps in the user messages but human-like datetime indicators. Don't center the user messages too much about your assumption of how the assistant would respond to the previous message, but more like you're just saying stuff to a dutiful assistant. Make sure the user_message contains all the crucial information relevant to the base_fact. The knowledge base is: {kb.model_dump_json()}.\n\n"

    print("Generating multiturn conversations...")
    multiturn_convos = get_model_response(MultiTurnConvos, prompt, GEMINI_PRO)

    if save:
        output_path = os.path.join(OUTPUT_PATH, CONVO_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "multiturn_convos.json")
        save_pydantic_to_json(multiturn_convos, file_path)
        print(f"Multiturn conversations saved to {file_path}")

    return multiturn_convos