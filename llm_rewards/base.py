from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class RewardOutput:
    """Output from reward computation."""
    values: torch.Tensor  # Shape: (batch_size,)
    info: Optional[Dict[str, Any]] = None


class RewardFunction:
    """Base class for all reward functions."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, **kwargs) -> RewardOutput:
        """Apply the reward function."""
        output = self.compute(**kwargs)
        output.values = output.values * self.weight
        return output

    def compute(self, **kwargs) -> RewardOutput:
        """Implement reward computation logic."""
        raise NotImplementedError("Reward functions must implement compute()")