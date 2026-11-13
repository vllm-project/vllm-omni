# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validated architecture of the official ZONOS3-Mini AR checkpoint.

Derived from zonos3.reference.lm.Zonos3LMConfig; see LICENSE.Zyphra.
Parameters are read from params.json, never inferred from a family default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ZONOS3_PROMPT_KEY = "zonos3_prompt"


@dataclass(frozen=True)
class Zonos3Architecture:
    n_layers: int
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    norm_eps: float
    rope_theta: float
    max_seqlen: int
    ffn_hidden: int
    moe_hidden: int
    moe_routed_dim: int
    n_experts: int
    text_vocab: int
    speaker_embedding_dim: int
    condition_bucket_counts: tuple[int, ...]
    condition_names: tuple[str, ...]
    condition_edges: dict[str, tuple[float, ...]]
    condition_units: dict[str, str]
    text_tokenizer_spec: dict[str, Any] | None = None
    text_extra_tokens: int = 0
    speaker_embedding_codec: str | None = None
    speaker_reference_prefix_enabled: bool = False
    speaker_reference_max_seconds: float = 30.0
    audio_codec: str = "opus_dual_3n_600k_sched600k_r4096_step600000"
    codebook_size: int = 4096
    n_codebooks: int = 10
    span_end_id: int | None = None
    loss_softcap: float = 15.0
    moe_topk: int = 2
    gdn_period: int = 4

    @property
    def audio_vocab(self) -> int:
        return self.codebook_size + 2 + (self.span_end_id is not None)

    @property
    def text_rows(self) -> int:
        return self.text_vocab + 1 + self.text_extra_tokens

    @property
    def eoa_id(self) -> int:
        return self.codebook_size

    @property
    def audio_pad_id(self) -> int:
        return self.codebook_size + 1

    @property
    def n_sources(self) -> int:
        return 2 * self.n_layers + 1

    @property
    def condition_count(self) -> int:
        return len(self.condition_bucket_counts)

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.n_codebooks

    def uses_gdn(self, layer: int) -> bool:
        return (layer + 1) % self.gdn_period != 0

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> Zonos3Architecture:
        p = params.get("model", params)
        required = {
            "n_layers": 11,
            "n_codebooks": 10,
            "codebook_size": 4096,
            "attention_pattern": "gdn3_global1",
            "residual_type": "full_attnres",
            "use_xsa": False,
            "speaker_enabled": True,
            "condition_tokens_enabled": True,
            "moe_router_type": "linear",
            "moe_router_use_eda": False,
            "moe_balancing_strategy": "quantile",
            "moe_router_topk": 2,
            "moe_start_from_layer": 1,
            "moe_end_from_layer": 0,
        }
        for key, wanted in required.items():
            if p.get(key) != wanted:
                raise ValueError(f"Unsupported ZONOS3-Mini architecture: {key}={p.get(key)!r}; expected {wanted!r}")
        optional = {
            "embedding_norm": True,
            "qknorm": True,
            "qwen_gating_style": "headwise",
            "flash_msa_every_n_layers": 0,
            "sliding_window": -1,
            "ffn_out_norm": False,
            "attn_out_norm": False,
            "moe_routed_outnorm": False,
            "use_eda": None,
            "rope_head_dim": None,
            "speaker_lda_dim": None,
            "speaking_rate_num_buckets": 0,
            "quality_num_buckets": 0,
            "speaker_background_token_enabled": False,
            "accurate_mode_token_enabled": False,
        }
        for key, wanted in optional.items():
            if key in p and p[key] != wanted and not (p[key] is None and not wanted):
                raise ValueError(f"Unsupported ZONOS3-Mini option: {key}={p[key]!r}")
        dim, head_dim = int(p["dim"]), int(p["head_dim"])
        heads, kv_heads = int(p["n_heads"]), int(p["n_kv_heads"])
        if min(dim, head_dim, heads, kv_heads) < 1 or head_dim % 2 or heads % kv_heads:
            raise ValueError("Invalid Zonos3 attention dimensions")
        if p.get("codebook_sizes", [4096] * 10) not in (None, [4096] * 10):
            raise ValueError("ZONOS3-Mini requires ten uniform 4096-entry codebooks")
        if (int(p["eoa_id"]), int(p["audio_pad_id"])) != (4096, 4097):
            raise ValueError("Unexpected Zonos3 EOA/padding IDs")
        span_end = p.get("span_end_id")
        if span_end not in (None, 4098):
            raise ValueError("Unexpected Zonos3 span-end ID")
        alpha = float(p["moe_alpha"])
        experts = int(p["moe_n_experts"])
        if not 0 < alpha <= 1 or experts < 3:
            raise ValueError("Invalid latent MoE dimensions or expert count")
        multiple = int(p.get("multiple_of", 256))
        moe_multiple = int(p.get("moe_multiple_of") or multiple)
        ffn_mult = float(p["ffn_dim_multiplier"])
        moe_mult = float(p.get("moe_ffn_dim_multiplier") or ffn_mult)
        if min(multiple, moe_multiple, ffn_mult, moe_mult) <= 0:
            raise ValueError("Invalid feed-forward dimensions")
        condition_count = int(p["condition_count"])
        names = tuple(str(name) for name in p.get("condition_names", ()))
        counts = tuple(
            int(n) for n in (p.get("condition_bucket_counts") or [int(p["condition_n_buckets"])] * condition_count)
        )
        if (
            names != ("loudness", "fmax", "speaking_rate")
            or condition_count != 3
            or len(counts) != 3
            or min(counts) < 2
        ):
            raise ValueError("ZONOS3-Mini requires loudness, fmax and speaking_rate condition tables")
        edges = {
            str(name): tuple(float(x) for x in values) for name, values in (p.get("condition_edges") or {}).items()
        }
        for name, values in edges.items():
            if (
                name not in names
                or len(values) != counts[names.index(name)] - 1
                or not all(math.isfinite(v) for v in values)
                or any(b <= a for a, b in zip(values, values[1:]))
            ):
                raise ValueError(f"Invalid condition bucket edges for {name}")
        spec = p.get("text_tokenizer_spec")
        text_vocab = int(p["text_vocab"])
        if spec is not None and (
            (spec.get("kind"), spec.get("offset"), spec.get("bos"), spec.get("eos")) != ("superbpe", 4, 2, 3)
            or int(spec.get("vocab_size", -1)) + 4 != text_vocab
        ):
            raise ValueError("Unsupported Zonos3 text tokenizer specification")
        eps, theta = float(p["norm_eps"]), float(p["rope_theta"])
        max_len, extra, speaker_dim = (
            int(p["max_seqlen"]),
            int(p.get("text_extra_tokens", 0) or 0),
            int(p["speaker_embedding_dim"]),
        )
        if min(eps, theta, max_len, text_vocab, speaker_dim) <= 0 or extra < 0:
            raise ValueError("Invalid Zonos3 normalization, context, vocabulary or speaker dimensions")
        return cls(
            n_layers=11,
            dim=dim,
            n_heads=heads,
            n_kv_heads=kv_heads,
            head_dim=head_dim,
            norm_eps=eps,
            rope_theta=theta,
            max_seqlen=max_len,
            ffn_hidden=multiple * ((int(ffn_mult * dim) + multiple - 1) // multiple),
            moe_hidden=moe_multiple * ((int(moe_mult * dim) + moe_multiple - 1) // moe_multiple),
            moe_routed_dim=max(1, round(dim * alpha)),
            n_experts=experts,
            text_vocab=text_vocab,
            speaker_embedding_dim=speaker_dim,
            condition_bucket_counts=counts,
            condition_names=names,
            condition_edges=edges,
            condition_units=dict(p.get("condition_units") or {}),
            text_tokenizer_spec=spec,
            text_extra_tokens=extra,
            span_end_id=span_end,
            speaker_embedding_codec=p.get("speaker_embedding_codec"),
            speaker_reference_prefix_enabled=bool(p.get("speaker_reference_prefix_enabled", False)),
            speaker_reference_max_seconds=float(p.get("speaker_reference_max_seconds", 30.0)),
            audio_codec=str(p.get("audio_codec", "")),
            loss_softcap=float(p["loss_softcap"]),
        )
