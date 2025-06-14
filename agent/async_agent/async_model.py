from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor

from typing import Optional, Union

from agent.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_STRONG_MODEL
from agent.schemas import ChatMessage, Role

def create_async_openai_client() -> AsyncOpenAI:
    """Create a new AsyncOpenAI client instance."""
    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

def create_async_instructor_client(async_openai_client: AsyncOpenAI = None):
    """Create a new async instructor client instance."""
    if async_openai_client is None:
        async_openai_client = create_async_openai_client()
    return instructor.from_openai(async_openai_client, mode=instructor.Mode.TOOLS)

# Initialize Async OpenAI client and the instructor client
ASYNC_CLIENT = create_async_openai_client()
ASYNC_INSTRUCTOR_CLIENT = create_async_instructor_client(ASYNC_CLIENT)

def _as_dict(msg: Union[ChatMessage, dict]) -> dict:
    """
    Accept either ChatMessage or raw dict and return the raw dict.

    Args:
        msg: A ChatMessage object or a raw dict.

    Returns:
        A raw dict.
    """
    return msg if isinstance(msg, dict) else msg.model_dump()

async def get_model_response(
        messages: Optional[list[ChatMessage]] = None,
        message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = OPENROUTER_STRONG_MODEL,
        schema: Optional[BaseModel] = None,
        async_client: Optional[AsyncOpenAI] = None,
        async_instructor_client = None,
) -> Union[str, BaseModel]:
    """
    Get a response from a model using OpenRouter asynchronously, with optional schema for structured output.

    Args:
        messages: A list of ChatMessage objects (optional).
        message: A single message string (optional).
        system_prompt: A system prompt for the model (optional).
        model: The model to use.
        schema: A Pydantic BaseModel for structured output (optional).
        async_client: Optional AsyncOpenAI client to use. If None, uses the global client.
        async_instructor_client: Optional async instructor client to use. If None, creates one from the AsyncOpenAI client.

    Returns:
        A string response from the model if schema is None, otherwise a BaseModel object.
    """
    if messages is None and message is None:
        raise ValueError("Either 'messages' or 'message' must be provided.")

    # Use provided clients or fall back to global ones
    if async_client is None:
        async_client = ASYNC_CLIENT
    if async_instructor_client is None:
        async_instructor_client = create_async_instructor_client(async_client)

    # Build message history
    if messages is None:
        messages = []
        if system_prompt:
            messages.append(_as_dict(ChatMessage(role=Role.SYSTEM, content=system_prompt)))
        messages.append(_as_dict(ChatMessage(role=Role.USER, content=message)))
    else:
        messages = [_as_dict(m) for m in messages]

    if schema is None:
        completion = await async_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content
    else: 
        completion = await async_instructor_client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"provider": {"require_parameters": True}},
            response_model=schema
        )
        return completion 