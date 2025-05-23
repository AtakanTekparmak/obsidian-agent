from openai import OpenAI

from pydantic import BaseModel
from typing import Optional, Union

from agent.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_STRONG_MODEL
from data.schemas.personas import Personas

# Initialize OpenAI client
CLIENT =  OpenAI(
    api_key = OPENROUTER_API_KEY,
    base_url = OPENROUTER_BASE_URL,
)

# Legend
"""
a.json = gemini = 6.5
b.json = opus   = 8.5
c.json = sonnet = 7.0
"""

def get_model_response(
        prompt: str,
        schema: BaseModel,
        model: str = "anthropic/claude-sonnet-4"
) -> BaseModel:
    """
    Get a response from a model using OpenRouter, with schema for structured output.

    Args:
        prompt: The user prompt
        schema: A Pydantic BaseModel for structured output.
        model: The model to use.

    Returns:
        A BaseModel object.
    """
    # Modify the propt to enforce the JSON schema
    addition = f"\n\nGive only the JSON output. Below is the schema for you to adhere to:\n {Personas.model_json_schema()}"
    prompt = prompt + addition

    # Get the raw response from model
    completion = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    response = completion.choices[0].message.content

    if "```json" in response and "```" in response:
        response = response.split("```json")[1].split("```")[0]

    return schema.model_validate_json(response)

prompt = """
Generate 8 personas for the scenario: Groningen, the Netherlands in 2025. Make sure to include a detailed backstory for each persona and try to make the relationships between the personas realistic. Also, in relationships, make sure to include other personas that are in the list of personas that you will generate. Make sure to include at least one relationship for each persona, but try to make the relationships realistic and not just random. Make sure to keep the backstories and relationships consistent with the scenario and keep the backstories detailed. Not every persona needs 2-3 relationships and not all of them need to be connected. Some personas can have no relationships, some can have 2-3 relationships, no restrictions. Make sure the backstory is detailed and captures the persona fully.
"""
response = get_model_response(
    prompt=prompt,
    schema=Personas
)

print(response.model_dump_json(indent=2))