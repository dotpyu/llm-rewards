from typing import List
import os
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
from trlx import train
from llm_rewards import (
    RewardModel,
    SimpleThinkReward,
    LengthReward,
    create_reward_fn
)


def create_config() -> TRLConfig:
    """Create training configuration."""
    return TRLConfig(
        train=TrainConfig(
            seq_length=512,
            epochs=100,
            total_steps=1000,
            batch_size=4,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            tracker="wandb",
            logging_dir="./logs",
            checkpoint_dir="./checkpoints",
        ),
        model=ModelConfig(
            model_path="Qwen/Qwen2.5-7B-Instruct-1M",
            num_layers_unfrozen=2,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path="Qwen/Qwen2.5-7B-Instruct-1M",
            truncation_side="right",
            padding_side="right",
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(
                lr=1.4e-5,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=1.0e-6,
            ),
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs=dict(
                T_max=1000,
                eta_min=1.4e-5,
            ),
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.1,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            scale_reward=True,
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=256,
                top_k=0,
                top_p=0.9,
                do_sample=True,
            ),
        ),
    )


def create_prompts() -> List[str]:
    """Create training prompts that encourage reasoning."""
    return [
        "Explain why quantum computing is different from classical computing. Think step by step.",
        "What makes a good leader? Analyze the key traits and provide reasoning.",
        "How would you solve this coding problem? Think through your approach carefully.",
        "Design a system to manage a library. Explain your design decisions with reasoning.",
        "Why do stars twinkle? Break down the scientific explanation.",
    ]


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create config
    config = create_config()

    # Initialize reward functions
    rewards = [
        RewardModel(
            "OpenAssistant/reward-model-deberta-v3-large",
            weight=1.0,
            device=device,
            batch_size=4,
        ),
        SimpleThinkReward(
            weight=0.5,
            min_words=20,
            max_words=200,
            required_keywords=["because", "therefore", "since", "reason"],
        ),
        LengthReward(
            target_length=512,
            weight=0.2,
            tokenizer="Qwen/Qwen2.5-7B-Instruct-1M",
        ),
    ]

    # Create reward function
    reward_fn = create_reward_fn(
        rewards=rewards,
        device=device,
        normalize=True,
        clip_range=4.0,
    )

    # Ensure directories exist
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # Start training
    trainer = train(
        reward_fn=reward_fn,
        prompts=create_prompts(),
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()