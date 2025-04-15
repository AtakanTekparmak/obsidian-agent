from google import genai
from google.genai.chats import Chat
from pydantic import BaseModel
from typing import Optional, Union

from agent.settings import GEMINI_API_KEY, GEMINI_FLASH

# Initialize the client
CLIENT = genai.Client(api_key=GEMINI_API_KEY)

def chat_with_model(
        prompt: str,
        chat: Chat = None,
        model: str = GEMINI_FLASH,
        schema: Optional[BaseModel] = None,
        ) -> tuple[Union[BaseModel, str], Chat]:
    """
    Chat with the model using Google's GenAI SDK.

    Args:
        prompt: The prompt to send to the model
        chat: The chat to use (optional, will create a new one if not provided)
        model: The model to use
        schema: The schema of the response (optional, will return the raw response if not provided)

    Returns:
        The structured response or the raw response and the chat object
    """
    if not chat:
        chat = CLIENT.chats.create(model=model)

    generation_config = None
    if schema:
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

    response = chat.send_message(
        message=prompt,
        config=generation_config,
    ) if generation_config else chat.send_message(message=prompt)

    returned_response = response.parsed if schema else response.text

    return returned_response, chat