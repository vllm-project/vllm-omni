# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration classes for Google Magenta RealTime 2 (magenta-realtime-2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "mrt2_base": {
        "encoder_dim": 1024,
        "temporal_num_layers": 20,
        "temporal_hidden_size": 3072,
        "temporal_ffn_dim": 8192,
        "temporal_num_heads": 24,
        "temporal_head_dim": 128,
        "temporal_self_past_horizon": 25,
        "temporal_cross_past_horizon": 25,
        "depth_num_layers": 6,
        "depth_hidden_size": 1024,
        "depth_ffn_dim": 4096,
        "depth_num_heads": 8,
        "depth_head_dim": 128,
    },
    "mrt2_small": {
        "encoder_dim": 256,
        "temporal_num_layers": 12,
        "temporal_hidden_size": 1024,
        "temporal_ffn_dim": 4096,
        "temporal_num_heads": 8,
        "temporal_head_dim": 128,
        "temporal_self_past_horizon": 41,
        "temporal_cross_past_horizon": 41,
        "depth_num_layers": 2,
        "depth_hidden_size": 768,
        "depth_ffn_dim": 3072,
        "depth_num_heads": 6,
        "depth_head_dim": 128,
    },
}


class MagentaRT2Config(PretrainedConfig):
    """Configuration for Magenta RealTime 2 (Temporal + Depthformer + SpectroStream)."""

    model_type = "magenta_realtime"

    def __init__(
        self,
        variant: str = "mrt2_base",
        checkpoint_filename: str | None = None,
        sample_rate: int = 48000,
        channels: int = 2,
        frame_rate_hz: float = 25.0,
        samples_per_frame: int = 1920,
        # Conditioning encoder
        encoder_num_channels: int = 144,
        num_mulan_channels: int = 12,
        mulan_vocab_per_channel: int = 1031,
        mulan_dim: int = 768,
        num_regular_channels: int = 132,
        regular_vocab_size: int = 1536,
        token_offset: int = 7,
        # SpectroStream target tokens
        num_codebooks: int = 12,
        codebook_size: int = 1024,
        num_extra_tokens: int = 6,
        vocab_size: int = 12294,
        logits_soft_cap: float = 30.0,
        # Attention defaults
        norm_eps: float = 1e-6,
        temporal_num_sinks: int = 1,
        depth_num_sinks: int = 0,
        depth_past_horizon: int = 12,
        # SpectroStream audio decoder
        ss_input_dim: int = 256,
        ss_num_quantizers: int = 64,
        ss_active_quantizers: int = 12,
        ss_codebook_size: int = 1024,
        ss_base_channels: int = 512,
        ss_ratios: tuple[tuple[int, int], ...] = (
            (1, 2),
            (1, 2),
            (1, 3),
            (1, 2),
            (1, 2),
            (2, 2),
            (2, 1),
        ),
        ss_mults: tuple[int, ...] = (2, 1, 2, 1, 1, 2, 1),
        ss_lookahead: int = 4,
        ss_stft_frame_size: int = 960,
        ss_stft_step_size: int = 480,
        ss_norm_factor: float = 1.0,
        # Sampling & MRv2 runtime knobs
        default_temperature: float = 1.1,
        default_top_k: int = 40,
        default_cfg_scale: float = 5.0,
        temporal_cuda_graphs: bool = True,
        depth_cuda_graphs: bool = True,
        spectrostream_cuda_graphs: bool = True,
        **kwargs: Any,
    ) -> None:
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unsupported MagentaRT2 variant {variant!r}; expected one of {sorted(VARIANT_CONFIGS)}")
        preset = VARIANT_CONFIGS[variant]
        self.variant = variant
        self.checkpoint_filename = checkpoint_filename or f"checkpoints/{variant}.safetensors"

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frame_rate_hz = float(frame_rate_hz)
        self.samples_per_frame = int(samples_per_frame)

        self.encoder_dim = int(kwargs.pop("encoder_dim", preset["encoder_dim"]))
        self.encoder_num_channels = int(encoder_num_channels)
        self.num_mulan_channels = int(num_mulan_channels)
        self.mulan_vocab_per_channel = int(mulan_vocab_per_channel)
        self.mulan_dim = int(mulan_dim)
        self.num_regular_channels = int(num_regular_channels)
        self.regular_vocab_size = int(regular_vocab_size)
        self.token_offset = int(token_offset)

        self.temporal_num_layers = int(kwargs.pop("temporal_num_layers", preset["temporal_num_layers"]))
        self.temporal_hidden_size = int(kwargs.pop("temporal_hidden_size", preset["temporal_hidden_size"]))
        self.temporal_ffn_dim = int(kwargs.pop("temporal_ffn_dim", preset["temporal_ffn_dim"]))
        self.temporal_num_heads = int(kwargs.pop("temporal_num_heads", preset["temporal_num_heads"]))
        self.temporal_head_dim = int(kwargs.pop("temporal_head_dim", preset["temporal_head_dim"]))
        self.temporal_self_past_horizon = int(
            kwargs.pop("temporal_self_past_horizon", preset["temporal_self_past_horizon"])
        )
        self.temporal_cross_past_horizon = int(
            kwargs.pop("temporal_cross_past_horizon", preset["temporal_cross_past_horizon"])
        )

        self.depth_num_layers = int(kwargs.pop("depth_num_layers", preset["depth_num_layers"]))
        self.depth_hidden_size = int(kwargs.pop("depth_hidden_size", preset["depth_hidden_size"]))
        self.depth_ffn_dim = int(kwargs.pop("depth_ffn_dim", preset["depth_ffn_dim"]))
        self.depth_num_heads = int(kwargs.pop("depth_num_heads", preset["depth_num_heads"]))
        self.depth_head_dim = int(kwargs.pop("depth_head_dim", preset["depth_head_dim"]))
        self.depth_past_horizon = int(depth_past_horizon)

        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.num_extra_tokens = int(num_extra_tokens)
        self.vocab_size = int(vocab_size)
        self.logits_soft_cap = float(logits_soft_cap)
        self.norm_eps = float(norm_eps)
        self.temporal_num_sinks = int(temporal_num_sinks)
        self.depth_num_sinks = int(depth_num_sinks)

        self.ss_input_dim = int(ss_input_dim)
        self.ss_num_quantizers = int(ss_num_quantizers)
        self.ss_active_quantizers = int(ss_active_quantizers)
        self.ss_codebook_size = int(ss_codebook_size)
        self.ss_base_channels = int(ss_base_channels)
        self.ss_ratios = tuple(tuple(int(v) for v in pair) for pair in ss_ratios)
        self.ss_mults = tuple(int(v) for v in ss_mults)
        self.ss_lookahead = int(ss_lookahead)
        self.ss_stft_frame_size = int(ss_stft_frame_size)
        self.ss_stft_step_size = int(ss_stft_step_size)
        self.ss_norm_factor = float(ss_norm_factor)

        self.default_temperature = float(default_temperature)
        self.default_top_k = int(default_top_k)
        self.default_cfg_scale = float(default_cfg_scale)
        self.temporal_cuda_graphs = bool(temporal_cuda_graphs)
        self.depth_cuda_graphs = bool(depth_cuda_graphs)
        self.spectrostream_cuda_graphs = bool(spectrostream_cuda_graphs)

        kwargs.setdefault("hidden_size", self.temporal_hidden_size)
        kwargs.setdefault("num_attention_heads", self.temporal_num_heads)
        kwargs.setdefault("num_hidden_layers", self.temporal_num_layers)
        kwargs.setdefault("architectures", ["MagentaRT2ForConditionalGeneration"])
        super().__init__(vocab_size=self.vocab_size, **kwargs)


def infer_magenta_rt2_variant(model_path: str | None) -> str | None:
    """Infer MagentaRT2 checkpoint variant (``mrt2_base`` / ``mrt2_small``) from model path or filename."""
    if not model_path:
        return None
    lower = model_path.lower()
    if "mrt2_small" in lower or "small" in lower:
        return "mrt2_small"
    if "mrt2_base" in lower or "magenta-realtime" in lower or "base" in lower:
        return "mrt2_base"
    return None


def resolve_magenta_rt2_checkpoint(
    model_path: str | None,
    config: MagentaRT2Config | None = None,
) -> Path | None:
    """Resolve the local Flax ``.safetensors`` checkpoint for a MagentaRT2 bundle or file."""
    if not model_path:
        return None
    root = Path(model_path)
    if root.is_file() and root.suffix == ".safetensors":
        return root
    variant = (
        (getattr(config, "variant", None) if config is not None else None)
        or infer_magenta_rt2_variant(model_path)
        or "mrt2_base"
    )
    for cand in (
        root / "checkpoints" / f"{variant}.safetensors",
        root / f"{variant}.safetensors",
        root / "checkpoints" / "mrt2_base.safetensors",
        root / "mrt2_base.safetensors",
    ):
        if cand.is_file():
            return cand
    return None


def magenta_rt2_allow_patterns(
    model_path: str | None,
    config: MagentaRT2Config | None = None,
) -> list[str]:
    """Return ``allow_patterns_overrides`` so vLLM discovers nested ``checkpoints/*.safetensors``."""
    ckpt = resolve_magenta_rt2_checkpoint(model_path, config)
    if ckpt is not None and model_path:
        root = Path(model_path)
        if root.is_dir():
            return [ckpt.relative_to(root).as_posix()]
    variant = (
        (getattr(config, "variant", None) if config is not None else None)
        or infer_magenta_rt2_variant(model_path)
        or "mrt2_base"
    )
    return [
        f"checkpoints/{variant}.safetensors",
        f"{variant}.safetensors",
        "checkpoints/*.safetensors",
        "*.safetensors",
    ]


__all__ = [
    "MagentaRT2Config",
    "VARIANT_CONFIGS",
    "infer_magenta_rt2_variant",
    "magenta_rt2_allow_patterns",
    "resolve_magenta_rt2_checkpoint",
]
