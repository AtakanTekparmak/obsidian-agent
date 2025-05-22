from openai import OpenAI

from pydantic import BaseModel
from typing import Optional, Union

from agent.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_STRONG_MODEL
from agent.schemas import ChatMessage, Role

# Initialize OpenAI client
CLIENT =  OpenAI(
    api_key = OPENROUTER_API_KEY,
    base_url = OPENROUTER_BASE_URL,
)

def _as_dict(msg: Union[ChatMessage, dict]) -> dict:
    """
    Accept either ChatMessage or raw dict and return the raw dict.

    Args:
        msg: A ChatMessage object or a raw dict.

    Returns:
        A raw dict.
    """
    return msg if isinstance(msg, dict) else msg.model_dump()

def get_model_response(
        messages: Optional[list[ChatMessage]] = None,
        message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = OPENROUTER_STRONG_MODEL,
        schema: Optional[BaseModel] = None,
) -> Union[str, BaseModel]:
    """
    Get a response from a model using OpenRouter, with optional schema for structured output.

    Args:
        messages: A list of ChatMessage objects (optional).
        message: A single messag    e string (optional).
        system_prompt: A system prompt for the model (optional).
        model: The model to use.
        schema: A Pydantic BaseModel for structured output (optional).

    Returns:
        A string response from the model if schema is None, otherwise a BaseModel object.
    """
    if messages is None and message is None:
        raise ValueError("Either 'messages' or 'message' must be provided.")

    # Build message history
    if messages is None:
        messages = []
        if system_prompt:
            messages.append(_as_dict(ChatMessage(role=Role.SYSTEM, content=system_prompt)))
        messages.append(_as_dict(ChatMessage(role=Role.USER, content=message)))
    else:
        messages = [_as_dict(m) for m in messages]

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
