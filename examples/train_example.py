
from typing import List
from trlx.data.default_configs import TRLConfig, default_ppo_config
from trlx import train
from llm_rewards import (
    RewardModel,
    SimpleThinkReward,
    LengthReward,
    create_reward_fn
)


def create_config():
    config = TRLConfig.load_yaml("trlx/configs/ppo_config.yaml")

    # Model settings
    config.model.model_path = "Qwen/Qwen1.5-0.5B"
    config.model.num_layers_unfrozen = 2

    # Tokenizer settings
    config.tokenizer.tokenizer_path = "Qwen/Qwen1.5-0.5B"
    config.tokenizer.truncation_side = "right"
    config.tokenizer.padding_side = "right"

    # Training settings
    config.train.batch_size = 4
    config.train.epochs = 100
    config.train.total_steps = 1000
    config.train.seq_length = 512
    config.train.checkpoint_interval = 1000
    config.train.eval_interval = 100
    config.train.pipeline = "PromptPipeline"
    config.train.trainer = "AcceleratePPOTrainer"

    # PPO settings
    config.method.name = "PPOConfig"
    config.method.num_rollouts = 128
    config.method.chunk_size = 128
    config.method.ppo_epochs = 4
    config.method.init_kl_coef = 0.1
    config.method.target = 6
    config.method.horizon = 10000
    config.method.gamma = 1
    config.method.lam = 0.95
    config.method.cliprange = 0.2
    config.method.cliprange_value = 0.2
    config.method.vf_coef = 0.1
    config.method.scale_reward = True
    config.method.ref_mean = None
    config.method.ref_std = None
    config.method.cliprange_reward = 10
    config.method.gen_kwargs = {
        "max_new_tokens": 256,
        "top_k": 0,
        "top_p": 0.9,
        "do_sample": True
    }

    # Optimizer settings
    config.optimizer.name = "adamw"
    config.optimizer.kwargs = dict(
        lr=1.4e-5,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=1.0e-6
    )

    # Scheduler settings
    config.scheduler.name = "cosine_annealing"
    config.scheduler.kwargs = dict(
        T_max=1000,
        eta_min=1.4e-5
    )

    return config


def create_prompts() -> List[str]:
    """Create training prompts that encourage reasoning."""
    return [
        "Explain why quantum computing is different from classical computing. Think step by step.",
        "What makes a good leader? Analyze the key traits and provide reasoning.",
        "How would you solve this coding problem? Think through your approach carefully.",
        "Design a system to manage a library. Explain your design decisions with reasoning.",
        "Why do stars twinkle? Break down the scientific explanation.",
        # Add more prompts that encourage reasoning
    ]


def main():
    config = create_config()

    rewards = [
        RewardModel(
            "OpenAssistant/reward-model-deberta-v3-large",
            weight=1.0,
            batch_size=4
        ),
        SimpleThinkReward(
            weight=0.5,
            min_words=20,
            max_words=200,
            required_keywords=["because", "therefore", "since", "reason"]
        ),
        LengthReward(
            target_length=512,
            weight=0.2
        )
    ]

    reward_fn = create_reward_fn(
        rewards=rewards,
        normalize=True,
        clip_range=4.0
    )

    trainer = train(
        reward_fn=reward_fn,
        prompts=create_prompts(),
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()