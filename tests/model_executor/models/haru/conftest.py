# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiny Haru fixtures: a checkpoint config, a codec small enough for CPU, a word-level tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.haru.configuration_haru import HaruCodecConfig, HaruHeadConfig, HaruSpec
from vllm_omni.model_executor.models.haru.vibevoice_codec import VibeVoiceCodec

PAD, EOP, EOS = 5, 6, 7
TINY_CODEC = HaruCodecConfig(
    latent_dim=4, base_channels=4, ratios=(2, 2), encoder_depths=(1, 1, 1), decoder_depths=(1, 1, 1)
)
TINY_SPEC = HaruSpec(
    sample_rate=16,
    frame_samples=4,
    latent_dim=4,
    pad_token_id=PAD,
    eop_token_id=EOP,
    dtype="float32",
    lm={"hidden_size": 8, "eos_token_id": EOS},
    codec=TINY_CODEC,
    head=HaruHeadConfig(hidden_size=8, latent_dim=4, num_layers=1, ffn_ratio=1.0),
)

# The released Haru-4B config.json, abridged to the fields the integration reads.
CHECKPOINT_CONFIG: dict[str, Any] = {
    "model_type": "haru",
    "architectures": ["HaruForConditionalGeneration"],
    "dtype": "bfloat16",
    "sample_rate": 24000,
    "frame_samples": 3200,
    "frame_rate": 7.5,
    "latent_dim": 64,
    "pad_token_id": 248076,
    "eop_token_id": 248071,
    "lm": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "tie_word_embeddings": True,
        "eos_token_id": 248044,
    },
    "codec": {"latent_dim": 64, "base_channels": 32, "ratios": [8, 5, 5, 4, 2, 2], "latent_std": 0.5},
    "head": {"hidden_size": 2560, "latent_dim": 64, "num_layers": 4, "guidance_range": [1.0, 2.0]},
}


def build_tiny_codec() -> VibeVoiceCodec:
    """A random tiny codec; its conv weights are allocated empty (they always come from a checkpoint)."""
    torch.manual_seed(0)
    codec = VibeVoiceCodec(TINY_CODEC)
    with torch.no_grad():
        for name, param in codec.named_parameters():
            if param.dim() > 1:
                param.normal_(0.0, 0.2)
            elif name.endswith("bias"):
                param.zero_()
    return codec.eval()


@pytest.fixture
def tiny_codec() -> VibeVoiceCodec:
    return build_tiny_codec()


def write_tokenizer(directory: Path) -> Path:
    """A word-level ``tokenizer.json`` (ids 0-15) next to the checkpoint config."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {w: i for i, w in enumerate(["<unk>", "<|im_start|>", "<|im_end|>", "system", "hi", "<pad>", "<eop>"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
    return directory


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    return write_tokenizer(tmp_path)
