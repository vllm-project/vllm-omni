# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime policy and deployment configuration for VibeVoice inference."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from numbers import Integral
from typing import Any

VIBEVOICE_DEFAULT_GUIDANCE_SCALE = 1.3
VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS = 10
VIBEVOICE_MIN_GUIDANCE_SCALE = 0.0
VIBEVOICE_MAX_GUIDANCE_SCALE = 20.0
VIBEVOICE_MAX_NUM_DIFFUSION_STEPS = 50
VIBEVOICE_MAX_DIFFUSION_GRAPH_BATCH_SIZE = 4
VIBEVOICE_RUNTIME_CONTROL_KEYS = frozenset(
    {
        "guidance_scale",
        "num_diffusion_steps",
    }
)


@dataclasses.dataclass(frozen=True)
class VibeVoiceRuntimeConfig:
    negative_kv_cache_memory_bytes: int = 4 * 1024**3
    negative_kv_activation_margin_bytes: int = 512 * 1024**2
    # Replay the fixed-step DPM denoising loop through manual CUDA
    # graphs (bitwise identical to eager). Disable to force the eager loop.
    diffusion_cuda_graph: bool = True
    # Pre-capture each configured active diffusion sub-batch before the first
    # request. None resolves to 1..max_num_seqs; an empty tuple disables only
    # startup warmup while preserving lazy runtime capture.
    diffusion_graph_warmup_batch_sizes: tuple[int, ...] | None = None
    # Replay the audio decode (acoustic decoder + semantic encoder +
    # projectors) through a per-request manual CUDA graph (bitwise identical
    # to eager). Shares a graph pool with the diffusion graph executor to
    # satisfy PyTorch's CUDACachingAllocator co-residency requirement.
    decode_cuda_graph: bool = True
    # Development/CI diagnostic: eligible graph capture failures normally fall
    # back to eager. Set this only when a validation run must prove that the
    # requested graph paths were actually captured.
    cuda_graph_capture_failure_fatal: bool = False

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> VibeVoiceRuntimeConfig:
        additional_config = getattr(vllm_config, "additional_config", None)
        raw = additional_config.get("vibevoice_runtime_config") if isinstance(additional_config, Mapping) else None
        if raw is None:
            return cls()
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, Mapping) and hasattr(raw, "__dict__"):
            raw = vars(raw)
        if not isinstance(raw, Mapping):
            raise ValueError("vibevoice_runtime_config must be a mapping.")

        field_types = {field.name: type(field.default) for field in dataclasses.fields(cls)}
        unknown_keys = sorted((key for key in raw if key not in field_types), key=str)
        if unknown_keys:
            raise ValueError(f"Unknown VibeVoice runtime config keys: {unknown_keys}")

        values: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "diffusion_graph_warmup_batch_sizes":
                if value is None:
                    values[key] = None
                    continue
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        "VibeVoice runtime config diffusion_graph_warmup_batch_sizes "
                        f"must be a list or tuple of positive integers, got {value!r}."
                    )
                batch_sizes: list[int] = []
                for batch_size in value:
                    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
                        raise ValueError(
                            "VibeVoice runtime config diffusion_graph_warmup_batch_sizes "
                            f"must contain only positive integers, got {batch_size!r}."
                        )
                    batch_sizes.append(int(batch_size))
                values[key] = tuple(batch_sizes)
                continue
            if field_types[key] is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"VibeVoice runtime config {key} must be a bool, got {value!r}.")
                values[key] = value
                continue
            if isinstance(value, bool):
                raise ValueError(f"VibeVoice runtime config {key} must be an integer, not bool.")
            if isinstance(value, Integral):
                values[key] = int(value)
                continue
            raise ValueError(f"VibeVoice runtime config {key} must be an integer, got {value!r}.")

        result = cls(**values)
        if result.negative_kv_cache_memory_bytes <= 0:
            raise ValueError("negative_kv_cache_memory_bytes must be positive.")
        if result.negative_kv_activation_margin_bytes < 0:
            raise ValueError("negative_kv_activation_margin_bytes must be non-negative.")
        return result

    def resolve_diffusion_graph_warmup_batch_sizes(
        self,
        max_num_seqs: int,
    ) -> tuple[int, ...]:
        """Resolve and validate the startup diffusion graph keys for one runner."""
        if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, Integral) or max_num_seqs <= 0:
            raise ValueError(f"VibeVoice max_num_seqs must be a positive integer, got {max_num_seqs!r}.")
        max_num_seqs = int(max_num_seqs)
        configured = self.diffusion_graph_warmup_batch_sizes
        if configured is None:
            return tuple(range(1, max_num_seqs + 1))
        batch_sizes = tuple(sorted(set(configured)))
        oversized = [batch_size for batch_size in batch_sizes if batch_size > max_num_seqs]
        if oversized:
            raise ValueError(
                f"VibeVoice diffusion graph warmup batch sizes {oversized} exceed max_num_seqs={max_num_seqs}."
            )
        return batch_sizes


__all__ = [
    "VIBEVOICE_DEFAULT_GUIDANCE_SCALE",
    "VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS",
    "VIBEVOICE_MAX_DIFFUSION_GRAPH_BATCH_SIZE",
    "VIBEVOICE_MAX_GUIDANCE_SCALE",
    "VIBEVOICE_MAX_NUM_DIFFUSION_STEPS",
    "VIBEVOICE_MIN_GUIDANCE_SCALE",
    "VIBEVOICE_RUNTIME_CONTROL_KEYS",
    "VibeVoiceRuntimeConfig",
]
