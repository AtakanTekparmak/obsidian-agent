from typing import Callable

# Export the RewardFunc type for use in rubrics
RewardFunc = Callable[..., float]

# Re-export everything from verifiers.verifiers for convenience
from verifiers.verifiers import *

__all__ = ["RewardFunc"]
