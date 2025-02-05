# LLM Rewards

A lean, modular reward functions for RLHF training with LLMs. Framework-agnostic design with built-in support for trlx, trl, and custom training loops.

## Install

```bash
pip install llm-rewards
```

## Quick Start

```python
from llm_rewards import RewardModel, SimpleThinkReward, LengthReward, XMLReward, create_reward_fn

# Create reward stack
rewards = [
    LengthReward(target_length=1024, weight=0.1),
    XMLReward(weight=0.5, partial_credit=True),
    RewardModel("your/reward/model", weight=1.0),
    SimpleThinkReward(weight=0.5) 
]

# Get framework-agnostic reward function
reward_fn = create_reward_fn(rewards, normalize=True)

# Use with trlx
from trlx import Trainer
trainer = Trainer(reward_fn=reward_fn)
trainer.train(...)
```

## Key Features

- Transformer reward models
- Reasoning validation (ThinkingReward)
- Length, format, XML validation
- Reference similarity
- Prompt relevance
- Framework adapters
- Batched inference
- Reward normalization

## Example Training Script

See `example/train_example.py` for full Qwen-2.5 0.5B training example.

## Custom Rewards

```python
from llm_rewards import RewardFunction, RewardOutput

class MyReward(RewardFunction):
    def compute(self, texts, **kwargs) -> RewardOutput:
        rewards = [score(text) for text in texts]
        return RewardOutput(values=torch.tensor(rewards))
```

## License
MIT
