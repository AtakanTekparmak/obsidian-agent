"""Environment registration for obsidian-retrieval environment."""

from skyrl_gym.envs import register

def register_obsidian_retrieval():
    """Register the obsidian-retrieval environment with SkyRL."""
    try:
        register(
            id="obsidian-retrieval",
            entry_point="training.retrieval.env:RetrievalEnv",
        )
        print("Successfully registered obsidian-retrieval environment")
    except Exception as e:
        print(f"Warning: Could not register environment: {e}")

# Register the environment when this module is imported
register_obsidian_retrieval() 