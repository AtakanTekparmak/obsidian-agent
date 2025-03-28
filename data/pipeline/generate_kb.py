import os
from dria import Prompt, DatasetGenerator, DriaDataset, Model
from dria.factory.persona import PersonaBackstory

from pydantic import BaseModel, Field

from data.settings import OUTPUT_PATH, PERSONAS_PATH, SCENARIOS, NUM_PERSONAS_PER_SCENARIO, STRONG_MODELS
from data.utils import save_data_to_json, save_dict_to_json

async def generate_personas() -> DriaDataset:
    """
    Generate a persona using Dria and save to JSON. Returns the DriaDataset object.

    Returns:
        DriaDataset: The DriaDataset object
    """
    # Create dataset
    dataset = DriaDataset(
        name="personas", 
        description="A dataset for personas", 
        schema=PersonaBackstory[-1].OutputSchema
    )

    # Create instructions
    instructions = [
        {"simulation_description": scenario, "num_of_samples": NUM_PERSONAS_PER_SCENARIO}
        for scenario in SCENARIOS
    ]

    # Create generator
    generator = DatasetGenerator(dataset=dataset)

    # Generate persona
    print("Generating personas...")
    await generator.generate(
        instructions=instructions,
        singletons=PersonaBackstory, 
        models=STRONG_MODELS
    )

    # Convert dataset to dataframe and then a list of dicts
    df = dataset.to_pandas()
    records: list[dict] = df.to_dict(orient="records")
    print(type(records))
    print(records)
    print(len(records))

    return dataset