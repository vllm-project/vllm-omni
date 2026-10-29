# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiny FastWAM-family components shared by the CPU unit tests.

``fill_deterministic`` sets every parameter from a seeded CPU generator in
sorted-name order, so the same weights can be rebuilt anywhere, including in
the script that produced the reference values in ``test_fastwam_model.py`` by
running the original FastWAM implementation on these weights.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from vllm_omni.diffusion.models.fastwam.config import PRESETS, ActionCodecConfig, FastWAMConfig
from vllm_omni.diffusion.models.fastwam.modeling.fastwam import FastWAMModel
from vllm_omni.diffusion.models.fastwam.modeling.umt5_encoder import WanUMT5Encoder
from vllm_omni.diffusion.models.fastwam.modeling.wan_vae_encoder import WanVAEFirstFrameEncoder

TINY_SUITES = ("suite_a", "suite_b")
TINY_CODEC = ActionCodecConfig(
    action_dim=7, action_horizon=8, num_tokens=2, latent_dim=4, decoder_hidden=8, num_suites=len(TINY_SUITES)
)
TINY_VAE = dict(base_dim=8, z_dim=48)
TINY_T5 = dict(vocab=32, dim=16, dim_attn=16, dim_ffn=24, num_heads=2, num_layers=2, num_buckets=8)
_TINY_SHAPES = dict(
    action_horizon=8,
    video_hidden_dim=16,
    video_ffn_dim=32,
    video_in_dim=48,
    video_out_dim=48,
    num_layers=2,
    num_heads=2,
    attn_head_dim=8,
    text_dim=16,
    freq_dim=16,
    camera_size=(32, 32),
    text_len=8,
    prompt_template="{task}",
    task_suites=TINY_SUITES,
    suite_ids={suite: i for i, suite in enumerate(TINY_SUITES)},
)


def tiny_cae_config() -> FastWAMConfig:
    """CAE-shaped tiny config: latent tokens decoded by a suite-conditioned codec."""
    return replace(
        PRESETS["cae_fastwam_libero"],
        variant="tiny_cae",
        action_hidden_dim=8,
        action_ffn_dim=16,
        action_token_dim=TINY_CODEC.latent_dim,
        action_codec=TINY_CODEC,
        **_TINY_SHAPES,
    )


def tiny_fastwam_config() -> FastWAMConfig:
    """Original-FastWAM-shaped tiny config: raw action tokens, no codec, shift 1."""
    return replace(
        PRESETS["fastwam_libero_uncond"],
        variant="tiny_fastwam",
        action_hidden_dim=8,
        action_ffn_dim=16,
        action_token_dim=7,
        **_TINY_SHAPES,
    )


def fill_deterministic(module: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
            value = 1.0 + 0.1 * noise if ("norm" in name or name.endswith("gamma")) else 0.2 * noise
            param.copy_(value.to(param.dtype))


def build_tiny_model(cfg: FastWAMConfig, seed: int = 0) -> FastWAMModel:
    model = FastWAMModel(
        cfg, vae_encoder=WanVAEFirstFrameEncoder(**TINY_VAE), text_encoder=WanUMT5Encoder(**TINY_T5)
    ).to(torch.float32)
    fill_deterministic(model, seed)
    return model.eval()


def tiny_inputs(cfg: FastWAMConfig, seed: int = 1) -> dict[str, torch.Tensor]:
    """uint8 image, umT5-shaped context, normalised proprio for :meth:`FastWAMModel.infer_action`."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    height, width = cfg.image_size
    return {
        "image": torch.randint(0, 256, (1, 3, height, width), generator=generator, dtype=torch.uint8),
        "context": torch.randn((1, 6, cfg.text_dim), generator=generator),
        "context_mask": torch.ones((1, 6), dtype=torch.bool),
        "proprio": 0.5 * torch.randn((1, cfg.proprio_dim), generator=generator),
    }


def _write_json(path, payload) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _wan_to_hf_umt5_key(name: str) -> str:
    from vllm_omni.diffusion.models.fastwam.modeling.umt5_encoder import _HF_LAYER_KEYS

    if name == "token_embedding.weight":
        return "shared.weight"
    if name == "norm.weight":
        return "encoder.final_layer_norm.weight"
    _, index, rest = name.split(".", 2)
    for layer, mapping in _HF_LAYER_KEYS.items():
        for hf_name, wan_name in mapping.items():
            if wan_name == rest:
                return f"encoder.block.{index}.layer.{layer}.{hf_name}"
    raise KeyError(name)


def write_tiny_assets(root, cfg: FastWAMConfig, model: FastWAMModel) -> tuple[str, str]:
    """Write a tiny FastWAM checkpoint dir and a tiny Wan2.2-Diffusers-like base dir.

    Returns ``(model_dir, base_model_dir)``. Weights are ``model``'s, stored the way the
    real assets store them (raw ``.pt`` checkpoint, Diffusers VAE, HF umT5, tokenizer).
    """
    import os

    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model_dir = os.path.join(str(root), "checkpoint")
    base_dir = os.path.join(str(root), "base")
    os.makedirs(model_dir, exist_ok=True)

    torch.save(
        {
            "mot": model.mot.state_dict(),
            "proprio_encoder": model.proprio_encoder.state_dict(),
            "step": 1,
            "torch_dtype": "torch.float32",
        },
        os.path.join(model_dir, cfg.policy_checkpoint),
    )
    if cfg.action_codec is not None:
        codec = cfg.action_codec
        decoder = {f"decoder.{k}": v for k, v in model.action_codec.state_dict().items()}
        encoder = {"encoder.input_proj.weight": torch.zeros(8, codec.action_dim)}  # training-only, must be ignored
        torch.save(
            {
                "config": {
                    "action_dim": codec.action_dim,
                    "action_horizon": codec.action_horizon,
                    "num_tokens": codec.num_tokens,
                    "latent_dim": codec.latent_dim,
                    "encoder_hidden": 8,
                    "decoder_hidden": codec.decoder_hidden,
                    "video_cond_dim": 0,
                    "variational": True,
                    "num_suites": codec.num_suites,
                },
                "model": {**encoder, **decoder},
                "epoch": 1,
            },
            os.path.join(model_dir, cfg.action_codec_checkpoint),
        )
    for i, suite in enumerate(cfg.task_suites):
        stats = {
            "state": {"default": {"global_min": [-1.0 - i] * cfg.proprio_dim, "global_max": [1.0] * cfg.proprio_dim}},
            "action": {
                "default": {
                    "global_min": [-0.5] * (cfg.action_dim - 1) + [0.0],
                    "global_max": [0.5 + i] * (cfg.action_dim - 1) + [1.0],
                }
            },
        }
        _write_json(os.path.join(model_dir, cfg.stats_file(suite)), stats)

    vae = model.vae_encoder
    _write_json(
        os.path.join(base_dir, "vae", "config.json"),
        {
            "_class_name": "AutoencoderKLWan",
            "base_dim": TINY_VAE["base_dim"],
            "z_dim": vae.z_dim,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "temperal_downsample": [False, True, True],
            "patch_size": 2,
            "is_residual": True,
            "attn_scales": [],
            "in_channels": 12,
            "latents_mean": vae.mean.tolist(),
            "latents_std": vae.inv_std.reciprocal().tolist(),
        },
    )
    vae_state = {k: v.contiguous() for k, v in vae.state_dict().items()}
    vae_state["decoder.conv_in.weight"] = torch.zeros(1)  # decoder weights must be ignored
    save_file(vae_state, os.path.join(base_dir, "vae", "diffusion_pytorch_model.safetensors"))

    t5 = TINY_T5
    _write_json(
        os.path.join(base_dir, "text_encoder", "config.json"),
        {
            "architectures": ["UMT5EncoderModel"],
            "d_model": t5["dim"],
            "d_kv": t5["dim_attn"] // t5["num_heads"],
            "d_ff": t5["dim_ffn"],
            "num_heads": t5["num_heads"],
            "num_layers": t5["num_layers"],
            "vocab_size": t5["vocab"],
            "relative_attention_num_buckets": t5["num_buckets"],
            "relative_attention_max_distance": 128,
            "feed_forward_proj": "gated-gelu",
            "layer_norm_epsilon": 1e-6,
        },
    )
    save_file(
        {_wan_to_hf_umt5_key(k): v.contiguous() for k, v in model.text_encoder.state_dict().items()},
        os.path.join(base_dir, "text_encoder", "model.safetensors"),
    )

    words = [
        "<pad>",
        "</s>",
        "<unk>",
        "pick",
        "place",
        "the",
        "bowl",
        "plate",
        "open",
        "drawer",
        "warm",
        "up",
        "policy",
    ]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(os.path.join(base_dir, "tokenizer"))
    return model_dir, base_dir
