import re
from typing import List, Optional
from xml.etree import ElementTree as ET

import torch

from ..base import RewardFunction, RewardOutput


class XMLReward(RewardFunction):
    """Reward for XML format validation and required tags."""

    def __init__(
            self,
            required_tags: Optional[List[str]] = None,
            weight: float = 1.0,
            partial_credit: bool = True
    ):
        super().__init__(weight)
        self.required_tags = required_tags
        self.partial_credit = partial_credit

    def compute(self, texts: List[str], **kwargs) -> RewardOutput:
        rewards = []
        info = {'malformed': 0, 'missing_tags': 0}

        for text in texts:
            reward = 1.0
            if not re.search(r'<[^>]+>', text):
                rewards.append(0.0)
                info['malformed'] += 1
                continue

            try:
                if not re.match(r'<[^>]+>', text.strip()):
                    text = f'<root>{text}</root>'
                ET.fromstring(text)

                if self.required_tags:
                    for tag in self.required_tags:
                        if not re.search(f'<{tag}[^>]*>.*?</{tag}>', text, re.DOTALL):
                            reward -= 1.0 / len(self.required_tags)
                            info['missing_tags'] += 1

            except ET.ParseError:
                if self.partial_credit:
                    reward *= 0.5
                    opens = re.findall(r'<([^/][^>]*)>', text)
                    closes = re.findall(r'</([^>]+)>', text)
                    if opens and closes:
                        matches = sum(1 for tag in opens if tag in closes)
                        reward += 0.2 * (matches / max(len(opens), len(closes)))
                else:
                    reward = 0.0
                info['malformed'] += 1

            rewards.append(max(0.0, reward))

        return RewardOutput(values=torch.tensor(rewards), info=info)
