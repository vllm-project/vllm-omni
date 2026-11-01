# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate native ComfyUI-layout H3 files without materializing their weights."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

MINIMAX_H3_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


@dataclass(frozen=True)
class MiniMaxH3SingleFileSpec:
    transformer_config: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "MiniMaxH3SingleFileSpec":
        """Read tensor descriptors only; reject incomplete/quantized DiTs early."""
        with safe_open(path, framework="pt", device="cpu") as source:
            shapes = {}
            for name in source.keys():
                tensor = source.get_slice(name)
                if tensor.get_dtype() not in {"BF16", "F32"}:
                    raise ValueError(f"H3 A1 does not support quantized or non-BF16/FP32 tensors: {name}")
                shapes[name] = tuple(tensor.get_shape())

        def shape(name: str, ndim: int) -> tuple[int, ...]:
            if name not in shapes:
                raise ValueError(f"H3 single-file checkpoint is missing {name}")
            value = shapes[name]
            if len(value) != ndim or any(size <= 0 for size in value):
                raise ValueError(f"invalid H3 tensor shape for {name}: {value}")
            return value

        def block_count(prefix: str) -> int:
            indices = set()
            for name in shapes:
                if name.startswith(prefix):
                    index = name[len(prefix) :].split(".", 1)[0]
                    if not index.isdigit():
                        raise ValueError(f"unsupported H3 block name: {name}")
                    indices.add(int(index))
            if not indices or sorted(indices) != list(range(len(indices))):
                raise ValueError(f"H3 {prefix} indices must be contiguous from zero")
            return len(indices)

        grid, width = shape("adaln_t_table", 2)
        if grid < 2:
            raise ValueError("H3 AdaLN curve must contain at least two rows")
        hidden, video_dim = shape("video_patch_proj.weight", 2)
        head_dim = shape("blocks.0.attn.q_norm.weight", 1)[0]
        qkv_dim = shape("blocks.0.attn.qkv_proj.weight", 2)[0]
        ffn = shape("blocks.0.mlp.fc2.weight", 2)[1]
        if video_dim % 4 or qkv_dim % (3 * head_dim):
            raise ValueError("H3 tensor shapes do not match native patch/QKV dimensions")
        config = {
            "_class_name": "MiniMaxH3DiTModel",
            "num_layers": block_count("blocks."),
            "token_refiner_num_layers": block_count("token_refiner.blocks."),
            "hidden_size": hidden,
            "num_attention_heads": qkv_dim // (3 * head_dim),
            "attention_head_dim": head_dim,
            "ffn_hidden_size": ffn,
            "latents_dim": video_dim // 4,
            "audio_latents_dim": shape("audio_patch_proj.weight", 2)[1],
            "patch_size": [1, 2, 2],
            "text_dim": shape("condition_proj.weight", 2)[1],
            "time_embed_dim": width,
            "adaln_curve_grid": grid,
            "adaln_out_features": 18 * hidden,
            "final_adaln_out_features": 2 * hidden,
            "rope_inv_freq_len": shape("rope.inv_freq", 1)[0],
        }
        expected = {
            "adaln_t_table": (grid, width),
            "rope.inv_freq": (config["rope_inv_freq_len"],),
            "token_refiner.final_norm.weight": (hidden,),
            "final_layer.norm.weight": (hidden,),
        }
        for prefix, inputs, outputs in (
            ("video_patch_proj", video_dim, hidden),
            ("audio_patch_proj", config["audio_latents_dim"], hidden),
            ("condition_proj", config["text_dim"], hidden),
            ("final_layer.video_out", hidden, video_dim),
            ("final_layer.audio_out", hidden, config["audio_latents_dim"]),
            ("final_layer.adaln_proj.linear", width, 2 * hidden),
        ):
            expected[f"{prefix}.weight"] = (outputs, inputs)
            expected[f"{prefix}.bias"] = (outputs,)
        for family, count in (
            ("blocks", config["num_layers"]),
            ("token_refiner.blocks", config["token_refiner_num_layers"]),
        ):
            for index in range(count):
                prefix = f"{family}.{index}"
                for suffix, dimensions in {
                    "norm1.weight": (hidden,),
                    "norm2.weight": (hidden,),
                    "attn.q_norm.weight": (head_dim,),
                    "attn.k_norm.weight": (head_dim,),
                    "attn.qkv_proj.weight": (qkv_dim, hidden),
                    "attn.out_proj.weight": (hidden, qkv_dim // 3),
                    "mlp.fc1.weight": (2 * ffn, hidden),
                    "mlp.fc2.weight": (hidden, ffn),
                }.items():
                    expected[f"{prefix}.{suffix}"] = dimensions
                if family == "blocks":
                    expected[f"{prefix}.adaln_proj.linear.weight"] = (18 * hidden, width)
                    expected[f"{prefix}.adaln_proj.linear.bias"] = (18 * hidden,)
        if "adaln_basis" in shapes or "adaln_mean" in shapes:
            if not {"adaln_basis", "adaln_mean"} <= shapes.keys():
                raise ValueError("H3 adapter basis requires both adaln_basis and adaln_mean")
            dense_width = shape("adaln_mean", 1)[0]
            expected["adaln_basis"] = (width, dense_width)
            expected["adaln_mean"] = (dense_width,)
        if missing := expected.keys() - shapes.keys():
            raise ValueError(f"H3 single-file checkpoint is missing tensors: {sorted(missing)}")
        if extra := shapes.keys() - expected.keys():
            raise ValueError(f"unsupported H3 single-file tensors: {sorted(extra)}")
        for name, dimensions in expected.items():
            if shapes[name] != dimensions:
                raise ValueError(f"H3 tensor shape mismatch for {name}: {shapes[name]} != {dimensions}")
        return cls(transformer_config=config)
