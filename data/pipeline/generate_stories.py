from typing import List
import os

from data.model import get_model_response
from data.settings import OUTPUT_PATH, STORIES_PATH, GEMINI_PRO
from data.utils import save_pydantic_to_json
from data.schemas.personas import Persona
from data.schemas.stories import MomentaryStories

def generate_momentary_stories(
        num_stories: int,
        personas: List[Persona],
        save: bool = True
    ) -> MomentaryStories:
    """
    Generate a list of momentary stories from a list of personas.

    Args:
        personas: A list of personas
        save: Whether to save the momentary stories to a file

    Returns:
        A list of momentary stories
    """

    prompt = f"Generate {num_stories} momentary stories from the following personas: {personas}. Make sure to include a timestamp for each story. Make sure to include {num_stories} stories for each persona. Make sure that the stories align with the persona's backstory and relationships and are feasible. Make sure that the stories are more factual narratives than writing that is too poetic/journaly/philosophical. The stories should capture facts and happenings in the life of the persona. The stories could be consisting of multiple lines, but should be a single story/happening."

    # Generate stories
    print("Generating stories...")
    response = get_model_response(MomentaryStories, prompt, GEMINI_PRO)

    if save:
        output_path = os.path.join(OUTPUT_PATH, STORIES_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "stories.json")
        save_pydantic_to_json(response, file_path)

    return response
    