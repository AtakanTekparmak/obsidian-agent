import os
from typing import Optional, Union

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Define model
MODEL = "google/gemini-2.5-flash-preview"

# Initialize OpenAI client
CLIENT =  OpenAI(
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url = os.getenv("OPENROUTER_BASE_URL"),
)

def get_model_response(
        message: str = None,
        system_prompt: Optional[str] = None,
        schema: Optional[BaseModel] = None,
        model: str = MODEL,
) -> Union[str, BaseModel]:
    """
    Get a response from the model.
    """

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    else:
        messages = [
            {"role": "user", "content": message}
        ]

    if schema is None:
        completion = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content
    else: 
        completion = CLIENT.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema
        )
        return completion.choices[0].message.parsed
