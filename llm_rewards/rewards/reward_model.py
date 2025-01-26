from typing import List, Optional, Union
import torch
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer
from ..base import RewardFunction, RewardOutput


class RewardModel(RewardFunction):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            weight: float = 1.0,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 8,
            max_length: int = 2048,
    ):
        super().__init__(weight)
        if isinstance(model, str):
            self.model = AutoModelForSequenceClassification.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

        self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def compute(
            self,
            texts: List[str],
            prompts: Optional[List[str]] = None,
            **kwargs
    ) -> RewardOutput:
        rewards = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_prompts = None if not prompts else prompts[i:i + self.batch_size]

            if batch_prompts:
                inputs = [f"{prompt}\n\nResponse: {text}" for prompt, text in zip(batch_prompts, batch_texts)]
            else:
                inputs = batch_texts

            with torch.no_grad():
                encoded = self.tokenizer(
                    inputs,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)
                reward_scores = outputs.logits.squeeze(-1)
                rewards.extend(reward_scores.cpu().tolist())

        return RewardOutput(
            values=torch.tensor(rewards),
            info={"model_name": self.model.config._name_or_path}
        )