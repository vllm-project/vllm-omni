# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen Image 2.1 diffusion model components.

Lazy by necessity: the pipeline imports the distributed VAE
(`vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage21`),
which in turn imports the plain VAE module from this package. Eager imports
here would make that a circular import whenever the distributed module is
imported first.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AutoencoderKLQwenImage21",
    "QwenImage21Pipeline",
    "QwenImage21Transformer2DModel",
    "get_qwen_image_21_post_process_func",
    "get_qwen_image_21_pre_process_func",
]


def __getattr__(name: str) -> Any:
    if name == "AutoencoderKLQwenImage21":
        from .autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

        return AutoencoderKLQwenImage21
    if name == "QwenImage21Transformer2DModel":
        from .qwen_image_21_transformer import QwenImage21Transformer2DModel

        return QwenImage21Transformer2DModel
    if name in (
        "QwenImage21Pipeline",
        "get_qwen_image_21_post_process_func",
        "get_qwen_image_21_pre_process_func",
    ):
        from .pipeline_qwen_image_21 import (
            QwenImage21Pipeline,
            get_qwen_image_21_post_process_func,
            get_qwen_image_21_pre_process_func,
        )

        return {
            "QwenImage21Pipeline": QwenImage21Pipeline,
            "get_qwen_image_21_post_process_func": get_qwen_image_21_post_process_func,
            "get_qwen_image_21_pre_process_func": get_qwen_image_21_pre_process_func,
        }[name]
    raise AttributeError(name)
