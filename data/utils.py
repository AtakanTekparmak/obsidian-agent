import json
from pathlib import Path
import os

from pydantic import BaseModel

from data.settings import OUTPUT_PATH, KB_PATH
from data.schemas.kb import KnowledgeBase
def save_pydantic_to_json(model: BaseModel, filepath: str) -> None:
    """
    Save a Pydantic model to a JSON file.

    Args:
        model: The Pydantic model to save
        filepath: The path where to save the JSON file

    Raises:
        IOError: If there's an error writing to the file
    """
    try:
        with open(filepath, "w") as f:
            f.write(model.model_dump_json(indent=2))
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")
        raise

def load_kb_from_json(filepath: str = os.path.join(OUTPUT_PATH, KB_PATH, "kb.json")) -> KnowledgeBase:
    """
    Load a KnowledgeBase object from a JSON file.

    Args:
        filepath: The path to the JSON file

    Returns:
        A KnowledgeBase object
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return KnowledgeBase.model_validate(data)
    except IOError as e:
        print(f"Error loading file {filepath}: {e}")
        raise
