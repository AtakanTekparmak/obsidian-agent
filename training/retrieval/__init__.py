from .dataset import create_kb_with_personas, generate_question_prompt, build_verifiers_dataset

# Only import env-related stuff if skyrl_gym is available
try:
    import skyrl_gym
    from .env import RetrievalEnv
    from . import register_env
    _SKYRL_AVAILABLE = True
except ImportError:
    _SKYRL_AVAILABLE = False

def get_retrieval_rubric():
    """Get the retrieval rubric for evaluation."""
    # This function seems to be referenced but not implemented
    # Implementing a basic version for now
    return "Evaluate the retrieval accuracy and relevance of the response."

__all__ = [
    "create_kb_with_personas",
    "generate_question_prompt",
    "build_verifiers_dataset",
    "get_retrieval_rubric",
]

# Only add env-related exports if skyrl_gym is available
if _SKYRL_AVAILABLE:
    __all__.append("RetrievalEnv")
