import os
import json

from data.schemas.convo import MultiTurnConvo, MultiTurnConvos
from data.settings import OUTPUT_PATH, CONVO_PATH

# Define constants
CONVOS_PATH = os.path.join(OUTPUT_PATH, CONVO_PATH, "convos.json")

def load_convos() -> MultiTurnConvos:
    """
    Load the convos from the convos.json file.

    Returns:
        MultiTurnConvos: The convos from the convos.json file.
    """
    try:
        with open(CONVOS_PATH, "r") as f:
            data = json.load(f)
        return MultiTurnConvos.model_validate(data)
    except Exception as e:
        raise Exception(f"Error loading convos from {CONVOS_PATH}: {e}")
