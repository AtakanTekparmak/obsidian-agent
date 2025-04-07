import json
from pathlib import Path
from pydantic import BaseModel

def save_pydantic_to_json(model: BaseModel, filepath: str | Path) -> None:
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
