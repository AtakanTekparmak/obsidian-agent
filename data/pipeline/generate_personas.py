import os

from data.model import get_model_response
from data.settings import OUTPUT_PATH, PERSONAS_PATH, OPENROUTER_SONNET
from data.utils import save_pydantic_to_json
from data.schemas.personas import Personas


def generate_personas(num_personas: int, scenario: str, save: bool = True) -> Personas:
    """
    Generate a list of personas.

    Args:
        num_personas: The number of personas to generate
        scenario: The scenario of the personas
        save: Whether to save the personas to a file

    Returns:
        A list of personas
    """
    prompt = f"Generate {num_personas} personas for the scenario:\n{scenario}\nMake sure to include a detailed backstory for each persona and try to make the relationships between the personas realistic. Also, in relationships, make sure to include other personas that are in the list of personas that you will generate. Make sure to include at least one relationship for each persona, but try to make the relationships realistic and not just random. Make sure to keep the backstories and relationships consistent with the scenario and keep the backstories detailed."

    # Generate personas
    print("Generating personas...")
    response = get_model_response(
        schema=Personas, prompt=prompt, model=OPENROUTER_SONNET
    )

    if save:
        output_path = os.path.join(OUTPUT_PATH, PERSONAS_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "personas.json")
        save_pydantic_to_json(response, file_path)

    return response
