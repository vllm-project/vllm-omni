"""Example: Collect RL rollout data from Qwen3-Omni Thinker.

This script demonstrates how to use vLLM-Omni's existing logprobs
mechanism to collect per-token log-probabilities for RL training
(e.g., GSPO/GRPO). No model code changes are needed — just set
SamplingParams(logprobs=1).

Usage (Thinker-only, text output):
    # Start the server with thinker-only config
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --omni \\
        --stage-config examples/online_serving/qwen3_omni/qwen3_omni_moe_thinking.yaml

    # Run this script
    python examples/rl/collect_thinker_rollout.py

Output format (per response):
    {
        "prompt": "What is 2+2?",
        "response_text": "The answer is 4.",
        "response_token_ids": [791, 4320, 374, 220, 19, 13],
        "log_probs": [-0.35, -0.12, -0.08, -1.20, -0.05, -0.42],
        "total_log_prob": -2.22
    }

These log_probs are the "old_log_probs" in GSPO/GRPO training:
    ratio = exp(new_log_prob - old_log_prob)
    loss = -advantage * ratio
"""

import json

from openai import OpenAI

# Connect to vLLM-Omni server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def collect_rollout(
    prompt: str,
    n_samples: int = 4,
    temperature: float = 0.8,
    max_tokens: int = 256,
) -> list[dict]:
    """Generate N responses for a prompt and collect log-probs.

    Args:
        prompt: The input prompt.
        n_samples: Number of responses per prompt (G in GRPO).
        temperature: Sampling temperature for exploration.
        max_tokens: Maximum response length.

    Returns:
        List of rollout data dicts, one per response.
    """
    response = client.completions.create(
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        prompt=prompt,
        n=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,  # Request per-token log-probs
    )

    rollouts = []
    for choice in response.choices:
        token_logprobs = choice.logprobs.token_logprobs  # list[float]
        tokens = choice.logprobs.tokens  # list[str]

        rollouts.append(
            {
                "prompt": prompt,
                "response_text": choice.text,
                "tokens": tokens,
                "log_probs": token_logprobs,
                "total_log_prob": sum(token_logprobs),
            }
        )

    return rollouts


def main():
    prompts = [
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms.",
        "Write a haiku about the ocean.",
    ]

    all_rollouts = []
    for prompt in prompts:
        rollouts = collect_rollout(prompt, n_samples=4)
        all_rollouts.extend(rollouts)
        print(f"Prompt: {prompt[:50]}...")
        for i, r in enumerate(rollouts):
            print(f"  Response {i}: log_prob={r['total_log_prob']:.2f}, len={len(r['tokens'])}")

    # Save for training
    # with open("thinker_rollout_data.json", "w") as f:
    #     json.dump(all_rollouts, f, indent=2)
    # print(f"\nSaved {len(all_rollouts)} rollouts to thinker_rollout_data.json")


if __name__ == "__main__":
    main()
