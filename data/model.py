from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import instructor

from typing import Optional, Union
from abc import ABC

from data.settings import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from agent.model import get_model_response as get_agent_response
from agent.async_agent import get_model_response as async_get_agent_response
from agent.schemas import ChatMessage, Role

def create_openai_client() -> OpenAI:
    """Create a new OpenAI client instance."""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

def create_async_openai_client() -> AsyncOpenAI:
    """Create a new AsyncOpenAI client instance."""
    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

def create_instructor_client(openai_client: OpenAI = None):
    """Create a new instructor client instance."""
    if openai_client is None:
        openai_client = create_openai_client()
    return instructor.from_openai(openai_client, mode=instructor.Mode.TOOLS)

def create_async_instructor_client(async_openai_client: AsyncOpenAI = None):
    """Create a new async instructor client instance."""
    if async_openai_client is None:
        async_openai_client = create_async_openai_client()
    return instructor.from_openai(async_openai_client, mode=instructor.Mode.TOOLS)

# Initialize the client
CLIENT = create_openai_client()

def get_model_response(
        prompt: str,
        model: str,
        schema: Optional[BaseModel] = None,
        client: Optional[OpenAI] = None
) -> BaseModel:
    """
    Get a response from a model using OpenRouter, with schema for structured output.

    Args:
        prompt: The user prompt
        schema: A Pydantic BaseModel for structured output.
        model: The model to use.
        client: Optional OpenAI client to use. If None, uses the global client.

    Returns:
        A BaseModel object.
    """
    if client is None:
        client = CLIENT
        
    # Modify the propt to enforce the JSON schema
    if schema is not None:
        addition = f"\n\nGive only the JSON output. Below is the schema for you to adhere to:\n {schema.model_json_schema()}"
        prompt = prompt + addition

    # Get the raw response from model
    completion = client.chat.completions.create(
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
            return get_model_response(prompt, model, schema, client)
    else:
        return response
    
class SFTModel(ABC):
    """
    Abstract class for an SFT model.
    """
    def __init__(self, num_turns: int):
        self.num_turns = num_turns
        self.messages: list[ChatMessage] = []
        # Each SFTModel instance gets its own clients to avoid bottlenecks
        self._client = create_openai_client()
        self._async_client = create_async_openai_client()
        
    def _add_message(self, message: Union[ChatMessage, dict]):
        """ Add a message to the conversation history. """
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
            
    def chat(self, message: Optional[str] = None) -> str:
        """ Chat with the LLM assistant using this instance's clients. """
        if message:
            self._add_message(ChatMessage(role=Role.USER, content=message))

        response = get_agent_response(
            messages=self.messages,
            client=self._client,
            use_vllm=False  # Data generation never uses vLLM, only OpenRouter
        )
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response))

        return response

    async def achat(self, message: Optional[str] = None) -> str:
        """Async chat with the LLM assistant using this instance's clients."""
        if message:
            self._add_message(ChatMessage(role=Role.USER, content=message))

        response = await async_get_agent_response(
            messages=self.messages,
            client=self._async_client,
            use_vllm=False  # Data generation never uses vLLM, only OpenRouter
        )
        self._add_message(ChatMessage(role=Role.ASSISTANT, content=response))

        return response