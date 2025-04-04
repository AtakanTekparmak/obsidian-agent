from typing import List
import enum
import os
import json
from pydantic import BaseModel

from data.model import get_model_response
from data.settings import OUTPUT_PATH, PERSONAS_PATH, GEMINI_PRO

class Gender(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"

class Relationship(BaseModel):
    name_surname: str
    relationship: str

class Birthplace(BaseModel):
    city: str
    country: str

class Persona(BaseModel):
    name_surname: str
    age: int
    gender: Gender
    birthplace: Birthplace
    occupation: str
    detailed_backstory: str
    relationships: List[Relationship]

class Personas(BaseModel):
    personas: List[Persona]

def generate_personas(
        num_personas: int, 
        scenario: str,
        save: bool = True
    ) -> List[Persona]:
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

    response = get_model_response(Personas, prompt, GEMINI_PRO)

    if save:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        filename = f"{OUTPUT_PATH}/{PERSONAS_PATH}/personas.json"
        try:
            with open(filename, "w") as f:
                json.dump(response.model_dump(), f, indent=2)
        except IOError as e:
            print(f"Error saving file {filename}: {e}")
            raise

    return response
