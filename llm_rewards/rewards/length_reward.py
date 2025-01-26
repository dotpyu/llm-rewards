from typing import List, Optional
import torch
from transformers import AutoTokenizer

from ..base import RewardFunction, RewardOutput
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from ..base import RewardFunction, RewardOutput


class LengthReward(RewardFunction):
    """Reward based on text length."""

    def __init__(
            self,
            target_length: int,
            weight: float = 1.0,
            tokenizer: Optional[str] = None,
            mode: str = "tokens"
    ):
        super().__init__(weight)
        self.target_length = target_length
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else None

    def compute(self, texts: List[str], **kwargs) -> RewardOutput:
        rewards = []

        for text in texts:
            if self.mode == "tokens" and self.tokenizer:
                length = len(self.tokenizer.encode(text))
            elif self.mode == "chars":
                length = len(text)
            else:
                length = len(text.split())

            # Gaussian penalty for deviation
            deviation = abs(length - self.target_length)
            reward = float(torch.exp(-0.5 * (deviation / self.target_length) ** 2))
            rewards.append(reward)

        return RewardOutput(values=torch.tensor(rewards))

