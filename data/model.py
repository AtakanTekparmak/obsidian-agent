from openai import OpenAI
from pydantic import BaseModel

from typing import Optional, Union
from abc import ABC

from data.settings import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from agent.model import get_model_response as get_agent_response
from agent.schemas import ChatMessage, Role

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
        try:
            return schema.model_validate_json(response) 
        except Exception as e:
            # If the response is not valid, try again
            return get_model_response(prompt, model, schema)
    else:
        return response
    
class SFTModel(ABC):
    """
    Abstract class for an SFT model.
    """
    def __init__(self, num_turns: int):
        self.num_turns = num_turns
        self.messages: list[ChatMessage] = []
        
    def _add_message(self, message: Union[ChatMessage, dict]):
        """ Add a message to the conversation history. """
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
            
    def chat(self, message: Optional[str] = None) -> str:
        """ Chat with the LLM assistant. """
        if message:
            self._add_message(ChatMessage(role=Role.USER, content=message))

        response = get_agent_response(messages=self.messages)
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response))

        return response