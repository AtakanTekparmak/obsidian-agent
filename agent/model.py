from openai import OpenAI
from pydantic import BaseModel
import instructor
import json
import re

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
    
    # For vLLM, we don't use Instructor at all - just return None
    if use_vllm:
        return None
    
    # Only use Instructor for non-vLLM backends
    return instructor.from_openai(openai_client, mode=instructor.Mode.TOOLS)

# Initialize OpenAI client and the instructor client
CLIENT = create_openai_client()
INSTRUCTOR_CLIENT = create_instructor_client(CLIENT)

def clean_and_parse_json(json_string: str) -> dict:
    """
    Clean and parse potentially malformed JSON from vLLM responses.
    Handles trailing characters and other common JSON formatting issues.
    """
    try:
        # First try parsing as-is
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Try to clean up common issues
        cleaned = json_string.strip()
        
        # Remove any trailing text after the JSON ends
        # Look for the last complete JSON object
        brace_count = 0
        last_valid_pos = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_valid_pos = i + 1
                    break
        
        if last_valid_pos > 0:
            cleaned = cleaned[:last_valid_pos]
        
        # Try parsing the cleaned version
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Last resort: try to extract JSON using regex
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, raise the original error with more context
            raise ValueError(f"Could not parse JSON response: {cleaned[:200]}...") from e

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
        instructor_client: Optional instructor client to use. Ignored if use_vllm=True.
        use_vllm: Whether to use vLLM backend instead of OpenRouter.

    Returns:
        A string response from the model if schema is None, otherwise a BaseModel object.
    """
    if messages is None and message is None:
        raise ValueError("Either 'messages' or 'message' must be provided.")

    # Use provided clients or fall back to global ones
    if client is None:
        client = CLIENT
    if instructor_client is None and not use_vllm:
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
            # For vLLM, use native guided JSON - no Instructor needed
            schema_dict = schema.model_json_schema()
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={
                    "guided_json": schema_dict,
                    "guided_decoding_backend": "outlines"
                }
            )
            
            # Parse the JSON response and create the schema object
            response_content = completion.choices[0].message.content
            response_json = clean_and_parse_json(response_content)
            return schema(**response_json)
        else:
            # For OpenRouter, use Instructor
            completion = instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"provider": {"require_parameters": True}},
                response_model=schema
            )
            return completion
