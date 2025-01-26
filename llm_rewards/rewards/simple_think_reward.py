import re
from typing import List, Optional
import torch

from ..base import RewardFunction, RewardOutput


class SimpleThinkReward(RewardFunction):
    """Reward for proper reasoning within <think> tags."""

    def __init__(
            self,
            weight: float = 1.0,
            min_words: int = 10,
            max_words: int = 100,
            required_keywords: Optional[List[str]] = None
    ):
        super().__init__(weight)
        self.min_words = min_words
        self.max_words = max_words
        # TODO: avoid hardcoding keywords
        self.required_keywords = required_keywords or ["because", "therefore", "since", "reason", "think", "consider"]

    def compute(self, texts: List[str], **kwargs) -> RewardOutput:
        rewards = []
        info = {
            'missing_tags': 0,
            'too_short': 0,
            'too_long': 0,
            'missing_keywords': 0
        }

        for text in texts:
            reward = 1.0

            # Extract content from <think> tags
            think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)

            if not think_matches:
                rewards.append(0.0)
                info['missing_tags'] += 1
                continue

            # Analyze all thinking segments
            for thinking in think_matches:
                thinking = thinking.strip()
                words = thinking.split()
                word_count = len(words)

                # Check length
                if word_count < self.min_words:
                    reward *= 0.5
                    info['too_short'] += 1
                elif word_count > self.max_words:
                    reward *= 0.7
                    info['too_long'] += 1

                # Check for reasoning keywords
                has_keywords = any(keyword in thinking.lower() for keyword in self.required_keywords)
                if not has_keywords:
                    reward *= 0.7
                    info['missing_keywords'] += 1

                # Check for complete sentences
                sentences = re.split(r'[.!?]+', thinking)
                valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
                if not valid_sentences:
                    reward *= 0.5

            rewards.append(max(0.0, reward))

        return RewardOutput(
            values=torch.tensor(rewards),
            info={
                **info,
                'avg_reward': sum(rewards) / len(rewards) if rewards else 0.0
            }
        )