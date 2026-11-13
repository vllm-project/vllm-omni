# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for the SANA-WM Stage-1 transformer.

Loaded from the standard diffusers ``transformer/config.json`` (a flat dict
keyed by the dataclass field names). The bespoke ``config.yaml`` release format
is handled once, offline, at checkpoint-conversion time — the runtime only ever
sees the converted, diffusers-standard config.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _as_tuple3(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return (1, int(value[0]), int(value[1]))
        if len(value) == 3:
            return (int(value[0]), int(value[1]), int(value[2]))
    return default


# LTX-2 VAE compression ratios (SANA-WM ships ``AutoencoderKLLTX2Video``).
# They live here rather than in the pipeline so the request-normalisation module
# can validate latent geometry without importing the pipeline.
SANA_WM_VAE_SPATIAL_COMPRESSION = 32
SANA_WM_VAE_TEMPORAL_COMPRESSION = 8


@dataclass(frozen=True)
class SanaWmConfig:
    """SANA-WM Stage-1 architecture and runtime config.

    Defaults mirror the first public HF release. ``from_json`` (reading
    ``transformer/config.json``) is the normal construction path.
    """

    architecture_name: str | None = "SanaMSVideoCamCtrl_1600M_P1_D20"
    num_blocks: int = 20
    hidden_size: int = 2240
    mlp_ratio: float = 3.0
    attn_type: str = "BidirectionalGDNTriton"
    softmax_every_n: int = 4
    linear_head_dim: int = 112
    conv_kernel_size: int = 4
    t_kernel_size: int = 3
    k_conv_only: bool = True
    ffn_type: str = "GLUMBConvTemp"
    pos_embed_type: str = "wan_rope"
    patch_size: tuple[int, int, int] = (1, 1, 1)
    qk_norm: bool = True
    cross_norm: bool = True
    mixed_precision: str = "bf16"
    fp32_attention: bool = True
    image_size: int = 720
    cam_attn_compress: int = 1
    use_chunk_plucker_post_attn: bool = True
    chunk_plucker_channels: int = 48
    chunk_plucker_post_attn_blocks: int = 20
    inference_flow_shift: float = 9.8
    scheduler_type: str = "flow_dpm-solver"
    chi_prompt: list[str] = field(default_factory=list)
    model_max_length: int = 300

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SanaWmConfig:
        """Build from a flat mapping of field name -> value.

        Unknown keys (e.g. the diffusers ``_class_name`` marker or the
        ``latent_channels`` / ``prompt_channels`` constructor kwargs) are
        ignored. ``patch_size`` is coerced from a JSON list back to a tuple.
        """
        field_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in data.items() if k in field_names}
        if "patch_size" in known:
            known["patch_size"] = _as_tuple3(known["patch_size"], cls.patch_size)
        return cls(**known)

    @classmethod
    def from_json(cls, path: str | Path) -> SanaWmConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data if isinstance(data, Mapping) else {})
