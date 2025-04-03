import asyncio
import glob
import os
from dria import Prompt, DatasetGenerator, DriaDataset
from pydantic import BaseModel, Field

from data.settings import OUTPUT_PATH, RELATIONSHIPS_PATH, STRUCTURED_OUTPUT_MODELS
from data.utils import save_data_to_json

# Define output schema
class Relationship(BaseModel):
    person_1: str = Field(..., title="Person 1")
    person_2: str = Field(..., title="Person 2")
    relationship: str = Field(..., title="Relationship")

class RelationshipsSchema(BaseModel):
    """
    A list of relationships between personas.
    """
    relationships: list[Relationship] = Field(..., title="Relationships")

async def generate_relationships(backstories: list[str]) -> list[str]:
    """
    Generate relationships between personas using Dria and save to JSON.
    Returns the list of backstories with relationships.

    Args:
        backstories: The list of backstories

    Returns:
        list[str]: The list of relationships
    """
    # Create dataset
    dataset = DriaDataset(
        name="relationships_for_personas_0", 
        description=f"A dataset for relationships between personas", 
        schema=RelationshipsSchema
    )
    
    # Create instructions
    instructions = [
        {"backstories": backstories}
    ]

    # Create prompt 
    prompter = Prompt(
        prompt="Generate relationships between the given personas. Make sure to generate plausible relationships by paying great attention to the backstories. Below are the backstories of the personas:\n {{backstories}}",
        schema=RelationshipsSchema
    )

    # Create generator
    generator = DatasetGenerator(dataset=dataset)

    # Generate relationships
    await generator.generate(
        instructions=instructions,
        singletons=prompter,
        models=STRUCTURED_OUTPUT_MODELS
    )

    # Convert dataset to dataframe and then a list of dicts
    df = dataset.to_pandas()
    relationships = df.to_dict(orient="records")

    # Determine the next file number
    existing_files = glob.glob(os.path.join(OUTPUT_PATH, RELATIONSHIPS_PATH, "relationships_*.json"))
    next_num = 0
    if existing_files:
        file_nums = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
        next_num = max(file_nums) + 1

    # Save the generated dataset with numbered filename
    filename = f"relationships_{next_num}.json"
    save_data_to_json(data_df=df, output_path=os.path.join(OUTPUT_PATH, RELATIONSHIPS_PATH), filename=filename)
    print(f"Dataset saved to {os.path.join(OUTPUT_PATH, RELATIONSHIPS_PATH, filename)}")

    return relationships