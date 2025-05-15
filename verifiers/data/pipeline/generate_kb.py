import os
from typing import List

from data.model import get_model_response
from data.settings import OUTPUT_PATH, KB_PATH, GEMINI_PRO
from data.utils import save_pydantic_to_json
from data.schemas.personas import Personas
from data.schemas.stories import MomentaryStories
from data.schemas.kb import (
    PersonaWithStories, 
    PersonasWithStories, 
    PersonalFact, 
    PersonalFacts, 
    KnowledgeBaseItem, 
    KnowledgeBase
)

def merge_stories_with_personas(
        stories: MomentaryStories,
        personas: Personas
    ) -> PersonasWithStories:
    """
    Merge stories with personas to create PersonaWithStories objects.

    Args:
        stories: MomentaryStories object
        personas: Personas object

    Returns:
        List of PersonaWithStories objects
    """
    persona_map = {persona.name_surname: persona for persona in personas.personas}
    merged_data: List[PersonaWithStories] = []

    for story_group in stories.stories:
        persona = persona_map.get(story_group.name_surname)
        if persona:
            merged_data.append(
                PersonaWithStories(
                    persona=persona,
                    stories=story_group.stories
                )
            )
        else:
            # Optional: Handle cases where a story group doesn't match any persona
            print(f"Warning: No persona found for name_surname '{story_group.name_surname}'")

    return PersonasWithStories(personas=merged_data)

    
def generate_kb(
        personas: Personas,
        stories: MomentaryStories,
        save: bool = True
    ) -> KnowledgeBase:
    """
    Generate a Knowledge Base (KB) from personas and stories.

    Args:
        personas: Personas object
        stories: MomentaryStories object

    Returns:
        KnowledgeBase object
    """
    personas_with_stories = merge_stories_with_personas(stories, personas)
    kb_items = []

    prompt = f"Generate a list of facts about the following personas: {personas_with_stories.personas}. The facts should be extracted from the stories and the personas. The facts should be in the following format: {PersonalFact.model_json_schema()}. Make sure to include facts about the persona's relationships and their interactions with other personas, as found in the stories. \n\n The facts should be extracted from the persona's fields other than the stories. These facts should be the 'initial facts' of the persona. The initial facts, then, can be either contractied or modified by the information in stories, which should be saved separately with a timestamp (the timestamp of the story it was extracted from) and text describing what changed/was discovered further. The only facts retrieved and saved from stories should be the facts that are absolutely relevant to the persona's backstory and/or initial facts and relationshop either in a way of negating them, changing their state or adding new information. Only save facts from relationships that are about important relationships like parent, partner, child, etc. \n\n Make sure to capture facts that you think are important, relevant to a persona's life as if you were their assistant and/or life coach. A handy way of saving 'state of things' facts could be the format key:value, but this format shouldn't be strictly enforced. \n\n"

    print("Generating personal facts...")
    personal_facts = get_model_response(PersonalFacts, prompt, GEMINI_PRO)

    persona_with_stories_map = {persona_with_stories.persona.name_surname: persona_with_stories for persona_with_stories in personas_with_stories.personas}

    kb_items = [
        KnowledgeBaseItem(
            persona_with_stories=persona_with_stories_map[fact.name_surname],
            facts=fact.facts
        ) for fact in personal_facts.facts
    ]

    kb = KnowledgeBase(items=kb_items)
    if save:
        output_path = os.path.join(OUTPUT_PATH, KB_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "kb.json")
        save_pydantic_to_json(kb, file_path)
        print(f"Knowledge base saved to {file_path}")

    return kb
    