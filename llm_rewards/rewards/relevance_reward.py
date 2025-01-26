from typing import List, Optional

import torch

from ..base import RewardFunction, RewardOutput


class RelevanceReward(RewardFunction):
    """Reward based on relevance to prompt."""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def compute(
            self,
            texts: List[str],
            prompts: Optional[List[str]] = None,
            **kwargs
    ) -> RewardOutput:
        if not prompts:
            raise ValueError("prompts required for relevance reward")

        if len(texts) != len(prompts):
            raise ValueError("texts and prompts must have same length")

        rewards = []
        for text, prompt in zip(texts, prompts):
            prompt_tokens = set(prompt.lower().split())
            if not prompt_tokens:
                rewards.append(0.0)
                continue

            text_tokens = set(text.lower().split())
            overlap = len(prompt_tokens & text_tokens)
            reward = overlap / len(prompt_tokens)
            rewards.append(reward)

        return RewardOutput(values=torch.tensor(rewards))

