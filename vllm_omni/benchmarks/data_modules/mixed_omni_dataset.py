# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Mixed Omni dataset for unified benchmark (i2t + t2i + it2i).

Generates a mixed list of ``MixedOmniSampleRequest`` objects that carry
per-request ``modalities`` and full chat messages, so all three task types
go through ``/v1/chat/completions`` (``--backend openai-chat-omni``) in a
single benchmark run. This enables DTPS mixed-task scheduling evaluation
within the unified ``vllm bench serve --omni`` framework.
"""

from __future__ import annotations

import base64
import io
import logging
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image
from vllm.benchmarks.datasets import SampleRequest

logger = logging.getLogger(__name__)

# Default prompts per task type
_DEFAULT_PROMPTS = {
    "i2t": "Describe this image in detail.",
    "t2i": "A serene mountain lake at dawn with mist rising over still waters.",
    "it2i": "Make the background a starry night sky.",
}

# Default it2i bot_task cycle
_DEFAULT_BOT_TASKS = ("recaption", "think", "think_recaption")

# Resolution distribution: 512x512 : 1024x1024 : 1280x720 = 1:1:1
_RESOLUTIONS = [(512, 512), (1024, 1024), (1280, 720)]


@dataclass
class MixedOmniSampleRequest(SampleRequest):
    """SampleRequest with per-request modalities + task type for mixed omni benchmark."""

    omni_chat_messages: list[dict[str, Any]] | None = None
    omni_extra_body: dict[str, Any] | None = None
    omni_task_type: str = ""


def _generate_placeholder_image(seed: int, size: tuple[int, int] = (512, 512)) -> bytes:
    """Generate a small PNG placeholder image."""
    rng = random.Random(seed)
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pick_resolution(seed: int) -> tuple[int, int]:
    """Pick a resolution from the 1:1:1 distribution deterministically."""
    rng = random.Random(seed)
    return rng.choice(_RESOLUTIONS)


def _build_i2t_messages(prompt: str, image_b64: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _build_t2i_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def _build_it2i_messages(prompt: str, image_b64: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _get_input_image_size(messages: list[dict[str, Any]] | None) -> str:
    """Extract input image size from messages (if any) by decoding the base64 PNG header."""
    if not messages:
        return "none"
    for msg in messages:
        for item in msg.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image/png;base64,"):
                    b64 = url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    return f"{img.size[0]}x{img.size[1]}"
    return "none"


def _log_sample_summary(idx: int, sample: MixedOmniSampleRequest) -> None:
    """Log a one-line summary of a sample request for debugging."""
    task = sample.omni_task_type
    prompt = sample.prompt[:60]
    eb = sample.omni_extra_body or {}
    out_size = f"{eb.get('width', '?')}x{eb.get('height', '?')}" if "width" in eb else "N/A"
    in_size = _get_input_image_size(sample.omni_chat_messages)
    modalities = eb.get("modalities", [])
    bot_task = eb.get("bot_task", "")
    bot_str = f" bot_task={bot_task}" if bot_task else ""
    logger.info(
        f"[MIXED-OMNI] #{idx:3d} type={task:4s} in_img={in_size:10s} "
        f"out_size={out_size:10s} modalities={modalities}{bot_str} "
        f'prompt="{prompt}"'
    )


def generate_mixed_omni_samples(
    config: dict[str, Any],
    *,
    seed: int = 0,
) -> list[MixedOmniSampleRequest]:
    """Generate a mixed list of i2t / t2i / it2i sample requests.

    Args:
        config: Dict with keys ``i2t``, ``t2i``, ``it2i`` (int counts).
        seed: RNG seed for reproducibility (image colors, resolution
              selection, and shuffle order are all deterministic).

    Returns:
        List of ``MixedOmniSampleRequest``. The list is shuffled using
        a seed-derived RNG so that task types are interleaved (not
        batched by type). Same seed always produces the same order.
    """
    num_i2t = int(config.get("i2t", 0))
    num_t2i = int(config.get("t2i", 0))
    num_it2i = int(config.get("it2i", 0))
    total = num_i2t + num_t2i + num_it2i
    if total == 0:
        raise ValueError("mixed-omni config must have at least one of i2t/t2i/it2i > 0")

    prompts_cfg = config.get("prompts", {})
    bot_tasks_cfg = config.get("bot_tasks", list(_DEFAULT_BOT_TASKS))

    samples: list[MixedOmniSampleRequest] = []

    # Generate i2t samples with mixed input resolutions
    for i in range(num_i2t):
        prompt = prompts_cfg.get("i2t", _DEFAULT_PROMPTS["i2t"])
        img_seed = seed * 1000 + i
        w, h = _pick_resolution(img_seed)
        img_bytes = _generate_placeholder_image(seed=img_seed, size=(w, h))
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        messages = _build_i2t_messages(prompt, img_b64)
        extra_body = {"modalities": ["text"], "temperature": 0}
        samples.append(
            MixedOmniSampleRequest(
                request_id=f"mixed-{i}",
                prompt=prompt,
                prompt_len=0,
                expected_output_len=256,
                omni_chat_messages=messages,
                omni_extra_body=extra_body,
                omni_task_type="i2t",
            )
        )

    # Generate t2i samples with mixed output resolutions
    for i in range(num_t2i):
        prompt = prompts_cfg.get("t2i", _DEFAULT_PROMPTS["t2i"])
        messages = _build_t2i_messages(prompt)
        img_seed = seed * 1000 + num_i2t + i
        w, h = _pick_resolution(img_seed)
        extra_body = {
            "modalities": ["image"],
            "temperature": 0,
            "width": w,
            "height": h,
        }
        samples.append(
            MixedOmniSampleRequest(
                request_id=f"mixed-{num_i2t + i}",
                prompt=prompt,
                prompt_len=0,
                expected_output_len=256,
                omni_chat_messages=messages,
                omni_extra_body=extra_body,
                omni_task_type="t2i",
            )
        )

    # Generate it2i samples with mixed input + output resolutions
    for i in range(num_it2i):
        prompt = prompts_cfg.get("it2i", _DEFAULT_PROMPTS["it2i"])
        img_seed = seed * 1000 + num_i2t + num_t2i + i
        w_in, h_in = _pick_resolution(img_seed)
        w_out, h_out = _pick_resolution(img_seed + 500)
        img_bytes = _generate_placeholder_image(seed=img_seed, size=(w_in, h_in))
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        messages = _build_it2i_messages(prompt, img_b64)
        bot_task = bot_tasks_cfg[i % len(bot_tasks_cfg)] if bot_tasks_cfg else None
        extra_body = {
            "modalities": ["image"],
            "temperature": 0,
            "width": w_out,
            "height": h_out,
        }
        if bot_task:
            extra_body["bot_task"] = bot_task
        samples.append(
            MixedOmniSampleRequest(
                request_id=f"mixed-{num_i2t + num_t2i + i}",
                prompt=prompt,
                prompt_len=0,
                expected_output_len=256,
                omni_chat_messages=messages,
                omni_extra_body=extra_body,
                omni_task_type="it2i",
            )
        )

    # Shuffle using a seed-derived RNG so task types are interleaved.
    # This is deterministic: same seed always produces the same order.
    shuffle_rng = random.Random(seed * 7919 + 1)
    shuffle_rng.shuffle(samples)

    # Log the full request sequence before sending
    logger.info("=" * 80)
    logger.info(f"[MIXED-OMNI] Request sequence (seed={seed}, total={total}):")
    for idx, sample in enumerate(samples):
        _log_sample_summary(idx, sample)
    logger.info("=" * 80)

    return samples
