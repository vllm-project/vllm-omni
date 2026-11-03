# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local random-weight Anima assets and real-checkpoint assets for common tests."""

import json
import tempfile
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import Unigram, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model, T5TokenizerFast

from vllm_omni.diffusion.models.anima.anima_text_conditioner import (
    ANIMA_TEXT_CONDITIONER_CONFIG,
    AnimaTextConditioner,
)
from vllm_omni.diffusion.models.anima.anima_transformer import ANIMA_TRANSFORMER_CONFIG, AnimaTransformer3DModel

CHECKPOINT_FILENAME = "anima.safetensors"


def tiny_anima_builder() -> str:
    """Build all components locally, without downloading any model weights or tokenizers."""
    model_dir = Path(tempfile.mkdtemp(prefix="tiny-anima-"))
    transformer_config = {
        **ANIMA_TRANSFORMER_CONFIG,
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "num_layers": 1,
        "mlp_ratio": 1.0,
        "text_embed_dim": 32,
        "adaln_lora_dim": 8,
        "max_size": (1, 64, 64),
    }
    conditioner_config = {
        **ANIMA_TEXT_CONDITIONER_CONFIG,
        "source_dim": 32,
        "target_dim": 32,
        "model_dim": 32,
        "num_layers": 1,
        "num_attention_heads": 2,
        "mlp_ratio": 1.0,
        "target_vocab_size": 8,
        "min_sequence_length": 8,
    }
    transformer = AnimaTransformer3DModel(**transformer_config).to(torch.bfloat16)
    conditioner = AnimaTextConditioner(**conditioner_config).to(torch.bfloat16)
    save_file(
        {
            **{f"transformer.{name}": tensor for name, tensor in transformer.state_dict().items()},
            **{f"text_conditioner.{name}": tensor for name, tensor in conditioner.state_dict().items()},
        },
        str(model_dir / CHECKPOINT_FILENAME),
    )
    (model_dir / "anima.json").write_text(
        json.dumps({"transformer": transformer_config, "text_conditioner": conditioner_config})
    )
    Qwen3Model(
        Qwen3Config(
            vocab_size=8,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
    ).to(torch.bfloat16).save_pretrained(model_dir / "text_encoder")
    AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    ).to(torch.bfloat16).save_pretrained(model_dir / "vae")
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "Dummy": 3, "prompt": 4}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(model_dir / "tokenizer")
    t5_tokenizer = Tokenizer(
        Unigram(
            [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0), ("Dummy", -1.0), ("prompt", -1.0)], unk_id=2
        )
    )
    T5TokenizerFast(tokenizer_object=t5_tokenizer, extra_ids=0).save_pretrained(model_dir / "t5_tokenizer")
    FlowMatchEulerDiscreteScheduler(shift=3.0).save_pretrained(model_dir / "scheduler")
    return str(model_dir)


def real_anima_model() -> str:
    """Keep cached real assets intact; return a disposable directory of links."""
    components = Path(
        snapshot_download(
            "circlestone-labs/Anima-Base-v1.0-Diffusers",
            allow_patterns=["text_encoder/*", "vae/*", "tokenizer/*", "t5_tokenizer/*", "scheduler/*"],
        )
    )
    checkpoint = hf_hub_download(
        "circlestone-labs/Anima", filename="split_files/diffusion_models/anima-base-v1.0.safetensors"
    )
    model_dir = Path(tempfile.mkdtemp(prefix="real-anima-"))
    (model_dir / CHECKPOINT_FILENAME).symlink_to(checkpoint)
    for name in ("text_encoder", "vae", "tokenizer", "t5_tokenizer", "scheduler"):
        if (components / name).is_dir():
            (model_dir / name).symlink_to(components / name, target_is_directory=True)
    return str(model_dir)
