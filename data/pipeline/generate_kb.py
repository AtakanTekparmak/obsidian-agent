import os

from data.model import get_model_response
from data.settings import OUTPUT_PATH, KB_PATH, OPENROUTER_SONNET
from data.utils import save_pydantic_to_json
from data.schemas.personas import Personas
from data.schemas.kb import (
    PersonalFact, 
    PersonalFacts, 
    KnowledgeBaseItem, 
    KnowledgeBase
)
    
def generate_kb(
        personas: Personas,
        save: bool = True
    ) -> KnowledgeBase:
    """
    Generate a Knowledge Base (KB) from personas.

    Args:
        personas: Personas object

    Returns:
        KnowledgeBase object
    """
    kb_items = []

    prompt = f"Generate a list of facts about the following personas: {personas}. The facts should be extracted from the personas. The facts should be in the following format: {PersonalFact.model_json_schema()}. Make sure to include facts about the persona's relationships. \n\n The facts should be extracted from the persona's fields. Only save facts from relationships that are about important relationships like parent, partner, child, etc. \n\n Make sure to capture facts that you think are important, relevant to a persona's life as if you were their assistant and/or life coach. A handy way of saving 'state of things' facts could be the format key:value, like age: 23, or birthplace: Amsterdam, the Netherlands. When saving relationships as facts, try to use a prolog-like syntax of father(X) if the persona is the father of X, for example. Each fact should be atomic and independent of other facts. \n\n"

    print("Generating personal facts...")
    personal_facts = get_model_response(
        schema=PersonalFacts,
        prompt=prompt,
        model=OPENROUTER_SONNET
    )

    persona_map = {persona.name_surname: persona for persona in personas.personas}

    kb_items = [
        KnowledgeBaseItem(
            persona=persona_map[fact.name_surname],
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
    