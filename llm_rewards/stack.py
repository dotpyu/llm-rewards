from typing import List

import torch

from .base import RewardFunction, RewardOutput


class RewardStack:
    """Combines multiple reward functions."""

    def __init__(self, rewards: List[RewardFunction]):
        """
        Args:
            rewards: List of reward functions to combine
        """
        self.rewards = rewards

    def __call__(self, **kwargs) -> RewardOutput:
        """
        Compute combined rewards.

        Args:
            texts: List[str], required
            prompts: Optional[List[str]]
            references: Optional[List[str]]
            **kwargs: Additional arguments passed to reward functions

        Returns:
            RewardOutput with combined rewards
        """
        rewards = []
        info = {}

        for i, reward in enumerate(self.rewards):
            output = reward(**kwargs)
            rewards.append(output.values)
            if output.info:
                info[f'reward_{i}'] = output.info

        total_reward = torch.stack(rewards).sum(dim=0)
        info['individual_rewards'] = [r.tolist() for r in rewards]
        return RewardOutput(values=total_reward, info=info)

    def get_weights(self) -> List[float]:
        """Get current reward weights."""
        return [r.weight for r in self.rewards]

    def set_weights(self, weights: List[float]) -> None:
        """Set reward weights."""
        if len(weights) != len(self.rewards):
            raise ValueError("Number of weights must match number of rewards")
        for reward, weight in zip(self.rewards, weights):
            reward.weight = weight