from openai import OpenAI
from pydantic import BaseModel

# Initialize the client
CLIENT = OpenAI()

def get_model_response(schema: BaseModel, prompt: str, model: str) -> BaseModel:
    """
    Get a structured response from the OpenAI model

    Args:
        schema: The schema of the response
        prompt: The prompt to send to the model
        model: The model to use

    Returns:
        The structured response
    """
    response = CLIENT.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ],
        text_format=schema
    )

    return response.output_parsed   