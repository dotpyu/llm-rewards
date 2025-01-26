from typing import List, Optional

import torch

from ..base import RewardFunction, RewardOutput


class SimilarityReward(RewardFunction):
    """Reward based on text similarity to reference."""

    def __init__(
            self,
            weight: float = 1.0,
            method: str = "token_overlap",
            threshold: float = 0.0
    ):
        super().__init__(weight)
        self.method = method
        self.threshold = threshold

    def compute(
            self,
            texts: List[str],
            references: Optional[List[str]] = None,
            **kwargs
    ) -> RewardOutput:
        if not references:
            raise ValueError("references required for similarity reward")

        if len(texts) != len(references):
            raise ValueError("texts and references must have same length")

        rewards = []
        for text, ref in zip(texts, references):
            if self.method == "token_overlap":
                text_tokens = set(text.lower().split())
                ref_tokens = set(ref.lower().split())
                if not ref_tokens:
                    rewards.append(0.0)
                    continue

                overlap = len(text_tokens & ref_tokens)
                score = overlap / len(ref_tokens)

            elif self.method == "char_ngram":
                # Character n-gram similarity
                n = 3
                text_ngrams = set(text[i:i + n] for i in range(len(text) - n + 1))
                ref_ngrams = set(ref[i:i + n] for i in range(len(ref) - n + 1))
                if not ref_ngrams:
                    rewards.append(0.0)
                    continue

                overlap = len(text_ngrams & ref_ngrams)
                score = overlap / len(ref_ngrams)

            else:
                raise ValueError(f"Unknown similarity method: {self.method}")

            rewards.append(float(score > self.threshold))

        return RewardOutput(values=torch.tensor(rewards))
