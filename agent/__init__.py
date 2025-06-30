"""
Obsidian Agent package.
"""

try:
    from .agent import Agent
    from .engine import ObsidianEngine
    from .model import get_model
    from .tools import get_tools
    from .utils import format_messages
    
    __all__ = [
        "Agent",
        "ObsidianEngine", 
        "get_model",
        "get_tools",
        "format_messages",
    ]
except ImportError:
    # If some modules can't be imported, just make the package importable
    __all__ = []
