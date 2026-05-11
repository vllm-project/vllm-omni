# SPDX-License-Identifier: Apache-2.0

"""Shared utility functions for the Raon module."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

# Weight suffixes that are safe to ignore when loading checkpoints.
IGNORE_WEIGHT_SUFFIXES: tuple[str, ...] = ("cluster_usage", "embed_sum", "initialized")


def _first_module_buffer(module: nn.Module) -> torch.Tensor | None:
    for _, buffer in module.named_buffers(recurse=True):
        return buffer
    return None


def _unwrap_singleton_tensor(value: Any) -> Any | None:
    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() != 1:
        return None
    return value.item()


def module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        buffer = _first_module_buffer(module)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")


def module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        buffer = _first_module_buffer(module)
        if buffer is not None:
            return buffer.dtype
        return torch.float32


def unwrap_singleton_list(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def collapse_exact_repeated_codec_snapshot(codes: torch.Tensor) -> torch.Tensor:
    """Collapse exact repeated [T, G] snapshots into one segment."""
    if codes.ndim != 2:
        return codes

    total_rows = int(codes.shape[0])
    if total_rows < 2:
        return codes

    max_repeats = min(total_rows, 16)
    for repeats in range(max_repeats, 1, -1):
        if total_rows % repeats != 0:
            continue
        seg_len = total_rows // repeats
        if seg_len <= 0:
            continue
        segment = codes[:seg_len]
        if all(torch.equal(segment, codes[i * seg_len : (i + 1) * seg_len]) for i in range(1, repeats)):
            return segment

    return codes


def coerce_optional_int(value: Any) -> int | None:
    value = unwrap_singleton_list(value)
    value = _unwrap_singleton_tensor(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_runtime_request_id(value: Any) -> str | None:
    value = unwrap_singleton_list(value)
    value = _unwrap_singleton_tensor(value)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    req_id = str(value).strip()
    return req_id if req_id else None


def cfg_get(config: object | None, key: str, default: Any = None) -> Any:
    """Get a value from a config object or mapping, with a default."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def get_placeholder_str(modality: str, _idx: int) -> str | None:
    """Return the placeholder string for a given modality."""
    from vllm_omni.tokenizers.raon_tokenizer import AUDIO_PLACEHOLDER_SEQ

    if modality.startswith("audio"):
        return AUDIO_PLACEHOLDER_SEQ
    raise ValueError("Only audio modality is supported")


def strip_raon_audio_markers(text: str) -> str:
    """Remove audio placeholder / pad tokens from decoded text."""
    from vllm_omni.tokenizers.raon_tokenizer import (
        AUDIO_END_TOKEN,
        AUDIO_INPUT_PAD_TOKEN,
        AUDIO_OUTPUT_PAD_TOKEN,
        AUDIO_START_TOKEN,
    )

    if not text:
        return ""
    for token in (
        AUDIO_START_TOKEN,
        AUDIO_END_TOKEN,
        AUDIO_INPUT_PAD_TOKEN,
        AUDIO_OUTPUT_PAD_TOKEN,
        "<|secondary_audio_pad|>",
        "<|audio_pad|>",
    ):
        text = text.replace(token, "")
    return " ".join(text.split())


def tracked_weight_loader(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    skip_prefixes: tuple[str, ...] = (),
    ignore_suffixes: tuple[str, ...] = (),
) -> tuple[set[str], dict[str, int]]:
    """Load weights via AutoWeightsLoader and return (loaded_names, seen_counts).

    seen_counts maps each weight name prefix (and '__total__') to the number of
    checkpoint tensors seen with that prefix.
    """
    loader = AutoWeightsLoader(
        model,
        skip_prefixes=list(skip_prefixes),
        ignore_unexpected_suffixes=list(ignore_suffixes),
    )
    seen_counts: dict[str, int] = defaultdict(int)

    def _tracked(w: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in w:
            seen_counts["__total__"] += 1
            yield name, tensor

    loaded = loader.load_weights(_tracked(weights))
    return loaded, seen_counts
