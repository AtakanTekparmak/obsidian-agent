import asyncio
from pydantic import BaseModel, Field
from dria import Prompt, DatasetGenerator, DriaDataset, Model
import pandas as pd

class PersonalStory(BaseModel):
    person_name: str = Field(..., title="Person's name")
    age: int = Field(..., title="Person's age")
    background: str = Field(..., title="Personal background")
    daily_activities: str = Field(..., title="Daily activities")
    relationships: str = Field(..., title="Relationships with others")
    past_events: str = Field(..., title="Past significant events")
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
    
    dataset = DriaDataset(
        name=f"personal_stories_{unique_id}",
        description="A dataset of diverse personal stories with temporal contexts and relationships",
        schema=PersonalStory
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
    personas = [
        {"profile": "tech professional in their 30s"},
        {"profile": "retired teacher in their 70s"},
        {"profile": "college student majoring in arts"},
        {"profile": "healthcare worker with family"},
        {"profile": "small business owner in a rural area"},
        {"profile": "urban single parent with two children"},
        {"profile": "international graduate student"},
        {"profile": "recently divorced middle-aged person"},
        {"profile": "young professional who just moved to a new city"},
        {"profile": "person managing a chronic health condition"},
        {"profile": "retiree starting a second career"},
        {"profile": "middle-aged person caring for elderly parents"},
    ]
    
    # Use a subset based on num_stories
    return personas[:num_stories]

async def generate_stories(num_stories: int = 10) -> pd.DataFrame:
    """
    Generate diverse personal stories using Dria SDK
    
    Args:
        num_stories (int): The number of stories to generate
    """
    # Create dataset
    dataset = create_story_dataset()
    
    # Create instructions
    instructions = generate_story_instructions(num_stories)
    
    # Define prompt for generating stories
    prompt_template = """
    Create a detailed personal profile for a {{profile}}. Include:
    
    1. A realistic name and age (age must be a number)
    2. Rich background information
    3. Their typical daily activities
    4. Their relationships with family, friends, colleagues, and others
    5. Significant past events that shaped them
    6. Their current life situation and challenges
    7. Their future plans and aspirations
    
    Make the story rich in temporal contexts (past, present, future) and include a
    variety of relationships that form a natural network. Balance factual information
    with subjective experiences and perspectives.
    
    Format your response to ensure all required fields are filled properly.
    """
    
    prompter = Prompt(prompt=prompt_template, schema=PersonalStory)
    
    # Create generator
    generator = DatasetGenerator(dataset=dataset)
    
    # Generate stories
    await generator.generate(
        instructions=instructions,
        singletons=prompter,
        models=Model.GPT4O
    )
    
    # Get the generated data
    df = dataset.to_pandas()
    
    # Validate and clean data
    df = clean_stories_data(df)
    
    return df

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
        'relationships', 'past_events', 'current_state', 'future_plans'
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
