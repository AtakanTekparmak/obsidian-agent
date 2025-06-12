from openai import OpenAI
from pydantic import BaseModel
import instructor

from typing import Optional, Union

from agent.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_STRONG_MODEL
from agent.schemas import ChatMessage, Role

def create_openai_client() -> OpenAI:
    """Create a new OpenAI client instance."""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

def create_vllm_client(host: str = "0.0.0.0", port: int = 8000) -> OpenAI:
    """Create a new vLLM client instance (OpenAI-compatible)."""
    return OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="EMPTY",  # vLLM doesn't require a real API key
    )

def create_instructor_client(openai_client: OpenAI = None, use_vllm: bool = False):
    """Create a new instructor client instance."""
    if openai_client is None:
        openai_client = create_openai_client()
    
    # For vLLM, we need to use JSON mode instead of tools mode
    mode = instructor.Mode.JSON if use_vllm else instructor.Mode.TOOLS
    return instructor.from_openai(openai_client, mode=mode)

# Initialize OpenAI client and the instructor client
CLIENT = create_openai_client()
INSTRUCTOR_CLIENT = create_instructor_client(CLIENT)

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
        client: Optional[OpenAI] = None,
        instructor_client = None,
        use_vllm: bool = False,
) -> Union[str, BaseModel]:
    """
    Get a response from a model using OpenRouter or vLLM, with optional schema for structured output.

    Args:
        messages: A list of ChatMessage objects (optional).
        message: A single message string (optional).
        system_prompt: A system prompt for the model (optional).
        model: The model to use.
        schema: A Pydantic BaseModel for structured output (optional).
        client: Optional OpenAI client to use. If None, uses the global client.
        instructor_client: Optional instructor client to use. If None, creates one from the OpenAI client.
        use_vllm: Whether to use vLLM backend instead of OpenRouter.

    Returns:
        A string response from the model if schema is None, otherwise a BaseModel object.
    """
    if messages is None and message is None:
        raise ValueError("Either 'messages' or 'message' must be provided.")

    # Use provided clients or fall back to global ones
    if client is None:
        client = CLIENT
    if instructor_client is None:
        instructor_client = create_instructor_client(client, use_vllm=use_vllm)

    # Build message history
    if messages is None:
        messages = []
        if system_prompt:
            messages.append(_as_dict(ChatMessage(role=Role.SYSTEM, content=system_prompt)))
        messages.append(_as_dict(ChatMessage(role=Role.USER, content=message)))
    else:
        messages = [_as_dict(m) for m in messages]

    if schema is None:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content
    else: 
        if use_vllm:
            # For vLLM, we don't use extra_body provider settings
            completion = instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=schema
            )
        else:
            # For OpenRouter, use the provider settings
            completion = instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"provider": {"require_parameters": True}},
                response_model=schema
            )
        return completion
