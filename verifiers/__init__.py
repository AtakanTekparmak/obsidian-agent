from typing import Callable

# Export the RewardFunc type for use in rubrics
RewardFunc = Callable[..., float]

__all__ = ["RewardFunc"]
