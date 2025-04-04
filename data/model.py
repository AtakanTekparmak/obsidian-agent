from google import genai
from pydantic import BaseModel

from data.settings import GEMINI_API_KEY, GEMINI_MODEL

# Initialize the client
CLIENT = genai.Client(api_key=GEMINI_API_KEY)

def get_model_response(schema: BaseModel, prompt: str, model: str = GEMINI_MODEL) -> BaseModel:
    """
    Get a structured response from the Gemini model

    Args:
        schema: The schema of the response
        prompt: The prompt to send to the model
        model: The model to use

    Returns:
        The structured response
    """
    response = CLIENT.models.generate_content(
        model=model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': schema,
        },
    )

    return response.parsed