import asyncio
from pydantic import BaseModel, Field
from dria import Prompt, DatasetGenerator, DriaDataset, Model
from dria.factory.persona import PersonaBio
import pandas as pd

class PersonalStory(BaseModel):
    person_name: str = Field(..., title="Person's name")
    age: int = Field(..., title="Person's age")
    background: str = Field(..., title="Personal background")
    daily_activities: str = Field(..., title="Daily activities")
    relationships: str = Field(..., title="Relationships with others")
    past_events: str = Field(..., title="Past significant events")
    chronological_timeline: str = Field(..., title="Chronological timeline of life events with dates")
    current_state: str = Field(..., title="Current life situation")
    future_plans: str = Field(..., title="Future plans and aspirations")

def create_story_dataset() -> DriaDataset:
    """
    Create a Dria dataset for personal stories
    
    Returns:
        DriaDataset: The created dataset
    """
    # Create a fresh dataset with a unique name to avoid mixing with existing data
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    
    # Use PersonaBio's schema for initial generation
    dataset = DriaDataset(
        name=f"personal_stories_{unique_id}",
        description="A dataset of diverse personal stories with temporal contexts and relationships",
        schema=PersonaBio[-1].OutputSchema
    )
    return dataset

def generate_story_instructions(num_stories: int = 10) -> list[dict]:
    """
    Generate instructions for diverse story creation
    
    Args:
        num_stories (int): The number of stories to generate
        
    Returns:
        list[dict]: The instructions for the stories
    """
    simulation_descriptions = [
        "A tech professional in their 30s working in a modern tech company",
        "A retired teacher in their 70s living in a suburban community",
        "A college student majoring in arts at a prestigious university",
        "A healthcare worker with family in a busy metropolitan area",
        "A small business owner in a rural area running a local shop",
        "An urban single parent with two children navigating city life",
        "An international graduate student adapting to a new country",
        "A recently divorced middle-aged person starting a new chapter",
        "A young professional who just moved to a new city for work",
        "A person managing a chronic health condition while maintaining an active lifestyle",
        "A retiree starting a second career in a different field",
        "A middle-aged person caring for elderly parents while balancing work",
    ]

    personas = [{"simulation_description": desc, "num_of_samples": 1} for desc in simulation_descriptions]
    
    # Use a subset based on num_stories
    return personas[:num_stories]

async def generate_stories(num_stories: int = 10) -> pd.DataFrame:
    """
    Generate diverse personal stories using Dria SDK
    
    Args:
        num_stories (int): The number of stories to generate
    """
    # Create dataset with PersonaBio schema
    dataset = create_story_dataset()
    
    # Create instructions
    instructions = generate_story_instructions(num_stories)
    
    # Create generator
    generator = DatasetGenerator(dataset=dataset)
    
    # Generate initial personas using PersonaBio
    await generator.generate(
        instructions=instructions,
        singletons=PersonaBio,
        models=[Model.DEEPSEEK_CHAT_OR]
    )
    
    # Get the generated data
    df = dataset.to_pandas()
    
    # Create a new dataset with our PersonalStory schema
    story_dataset = DriaDataset(
        name=f"personal_stories_enriched_{dataset.name}",
        description="Enriched personal stories with chronological timeline",
        schema=PersonalStory
    )
    
    # Create a new generator for the enriched dataset
    story_generator = DatasetGenerator(dataset=story_dataset)
    
    # Define prompt for generating stories
    prompt_template = """
    Create a detailed personal profile based on this backstory: {{bio}}. Include:
    
    1. A realistic name and age (age must be a number)
    2. Rich background information
    3. Their typical daily activities
    4. Their relationships with family, friends, colleagues, and others
    5. Significant past events that shaped them
    6. A detailed chronological timeline of their life events with specific dates (YYYY-MM-DD format)
    7. Their current life situation and challenges
    8. Their future plans and aspirations
    
    Make the story rich in temporal contexts (past, present, future) and include a
    variety of relationships that form a natural network. Balance factual information
    with subjective experiences and perspectives.
    
    For the chronological timeline, include at least 5-7 significant life events with dates,
    starting from early life and progressing to the present. Each event should be on a new line
    and include both the date and a brief description.
    
    Format your response to ensure all required fields are filled properly.
    """
    
    prompter = Prompt(prompt=prompt_template, schema=PersonalStory)
    
    # Generate enriched stories using the bio from PersonaBio
    await story_generator.generate(
        instructions=df.to_dict('records'),
        singletons=prompter,
        models=Model.GPT4O
    )
    
    # Get the final enriched data
    enriched_df = story_dataset.to_pandas()
    
    # Validate and clean data
    enriched_df = clean_stories_data(enriched_df)
    
    return enriched_df

def clean_stories_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate story data
    
    Args:
        df (pd.DataFrame): The dataframe to clean
    """
    # Drop rows with NaN values in critical fields
    df = df.dropna(subset=['person_name', 'age'])
    
    # Convert age to integer if possible
    df['age'] = df['age'].astype(float)
    
    # Ensure we have expected fields
    expected_fields = [
        'person_name', 'age', 'background', 'daily_activities', 
        'relationships', 'past_events', 'chronological_timeline',
        'current_state', 'future_plans'
    ]
    
    # Only keep expected fields
    df = df[expected_fields]
    
    return df

def get_stories(num_stories: int = 10) -> pd.DataFrame:
    """
    Run the story generation and return the dataset
    
    Args:
        num_stories (int): The number of stories to generate
    """
    df = asyncio.run(generate_stories(num_stories))
    return df
