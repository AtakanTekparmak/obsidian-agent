from openai import OpenAI
from pydantic import BaseModel

from typing import Optional

from data.settings import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

# Initialize the client
CLIENT =  OpenAI(
    api_key = OPENROUTER_API_KEY,
    base_url = OPENROUTER_BASE_URL,
)

def get_model_response(
        prompt: str,
        model: str,
        schema: Optional[BaseModel] = None
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
    if schema is not None:
        addition = f"\n\nGive only the JSON output. Below is the schema for you to adhere to:\n {schema.model_json_schema()}"
        prompt = prompt + addition

    # Get the raw response from model
    completion = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    response = completion.choices[0].message.content

    if "```json" in response and "```" in response:
        response = response.split("```json")[1].split("```")[0]

    if schema is not None:
        return schema.model_validate_json(response) 
    else:
        return response