"""Built-in reward functions."""

import re
from typing import List

import torch

from ..base import RewardFunction, RewardOutput


class FormatReward(RewardFunction):
    """Reward based on text format matching."""

    def __init__(
            self,
            pattern: str,
            weight: float = 1.0,
            case_sensitive: bool = False
    ):
        super().__init__(weight)
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(pattern, flags)

    def compute(self, texts: List[str], **kwargs) -> RewardOutput:
        rewards = [
            float(bool(self.pattern.match(text)))
            for text in texts
        ]
        return RewardOutput(values=torch.tensor(rewards))