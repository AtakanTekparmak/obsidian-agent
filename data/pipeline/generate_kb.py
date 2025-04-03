import os
import glob
from dria import DatasetGenerator, DriaDataset
from dria.factory.persona import PersonaBackstory

from data.settings import OUTPUT_PATH, PERSONAS_PATH, NUM_PERSONAS_PER_SCENARIO, STRONG_MODELS
from data.utils import save_data_to_json

async def generate_personas(scenario: str) -> list[str]:
    """
    Generate a persona using Dria and save to JSON. Returns the list of backstories.

    Returns:
        list[str]: The list of backstories
    """
    # Create dataset
    dataset = DriaDataset(
        name="personas_for_scenario_0", 
        description=f"A dataset for personas based on scenario: {scenario}", 
        schema=PersonaBackstory[-1].OutputSchema
    )

    # Create instructions with single scenario
    instructions = [
        {"simulation_description": scenario, "num_of_samples": NUM_PERSONAS_PER_SCENARIO}
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
    backstories = [persona["backstory"] for persona in records]
    

    # Determine the next file number
    existing_files = glob.glob(os.path.join(OUTPUT_PATH, PERSONAS_PATH, "personas_*.json"))
    next_num = 0
    if existing_files:
        file_nums = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
        next_num = max(file_nums) + 1

    # Save the generated dataset with numbered filename
    filename = f"personas_{next_num}.json"
    save_data_to_json(data_df=df, output_path=os.path.join(OUTPUT_PATH, PERSONAS_PATH), filename=filename)
    print(f"Dataset saved to {os.path.join(OUTPUT_PATH, PERSONAS_PATH, filename)}")

    return backstories