# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


def build_x_to_text_prompt(
    model: str,
    prompt: str,
    has_image: bool,
) -> tuple[dict[str, Any], list[int] | None]:
    """Build HunyuanImage-3 prompt tokens and stopping rules for T2T/I2T."""
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
        resolve_sys_type,
    )

    task = "i2t" if has_image else "t2t"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    build_kwargs: dict[str, Any] = {"task": task, "bot_task": None}
    if has_image:
        build_kwargs["num_images"] = 1
    built = build_prompt_tokens(prompt, tokenizer, **build_kwargs)
    return (
        {
            "prompt": prompt,
            "prompt_token_ids": built.token_ids,
            "modalities": ["text"],
            "use_system_prompt": resolve_sys_type(None),
        },
        resolve_stop_token_ids(task=task, bot_task=None, tokenizer=tokenizer, image_size="auto"),
    )
