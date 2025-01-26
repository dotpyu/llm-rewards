from typing import Callable, List, Optional, Dict, Any
import torch
from .stack import RewardStack
from .base import RewardFunction


class RLHFAdapter:
    """Generic adapter for RLHF frameworks."""
    def __init__(
            self,
            rewards: List[RewardFunction],
            device: str = "cuda",
            normalize: bool = True,
            clip_range: Optional[float] = 1.0,
            batch_size: int = 8
    ):
        self.reward_stack = RewardStack(
            rewards=rewards,
            normalize=normalize,
            clip_range=clip_range
        ).to(device)
        self.device = device
        self.batch_size = batch_size

    def __call__(
            self,
            prompts: List[str],
            responses: List[str],
            return_dict: bool = False,
            **kwargs
    ) -> Any:
        """
        Generic reward computation interface.

        Args:
            prompts: Input prompts
            responses: Generated responses
            return_dict: If True, returns dict with rewards and metadata
            **kwargs: Additional arguments passed to reward functions

        Returns:
            torch.Tensor or Dict: Reward values or dict with rewards and metadata
        """
        outputs = self.reward_stack(
            texts=responses,
            prompts=prompts,
            **kwargs
        )

        if return_dict:
            return {
                "rewards": outputs.values,
                "metadata": outputs.info or {}
            }
        return outputs.values

    @torch.no_grad()
    def batch_score(
            self,
            prompts: List[str],
            responses: List[str],
            **kwargs
    ) -> torch.Tensor:
        """Score responses in batches to save memory."""
        all_rewards = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_responses = responses[i:i + self.batch_size]

            rewards = self(
                prompts=batch_prompts,
                responses=batch_responses,
                **kwargs
            )
            all_rewards.append(rewards)

        return torch.cat(all_rewards)


def create_reward_fn(
        rewards: List[RewardFunction],
        **kwargs
) -> Callable:
    """
    Create a reward function compatible with various RLHF frameworks.

    Args:
        rewards: List of reward functions
        **kwargs: Additional arguments for the adapter

    Returns:
        Callable reward function in the format expected by the framework

    TODO: Add explicit support for existing RLHF frameworks
    """
    adapter = RLHFAdapter(rewards=rewards, **kwargs)
    return lambda p, r, **k: adapter(prompts=p, responses=r)