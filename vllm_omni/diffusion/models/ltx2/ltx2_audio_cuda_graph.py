# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime-owned CUDA Graph replay for the LTX2 audio Transformer."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LTX2AudioCUDAGraphConfig:
    """Model-local configuration parsed from ``additional_config``."""

    enabled: bool = False
    max_entries: int = 4

    @classmethod
    def from_additional_config(cls, additional_config: Mapping[str, Any] | None) -> LTX2AudioCUDAGraphConfig:
        if additional_config is None:
            return cls()
        if not isinstance(additional_config, Mapping):
            raise TypeError("additional_config must be a mapping or None")

        if "ltx2_audio_cuda_graph" not in additional_config:
            return cls()
        raw = additional_config["ltx2_audio_cuda_graph"]
        if not isinstance(raw, Mapping):
            raise TypeError("additional_config.ltx2_audio_cuda_graph must be a mapping")

        unknown = set(raw) - {"enabled", "max_entries"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown LTX2 audio CUDA Graph option(s): {names}")

        enabled = raw.get("enabled", False)
        max_entries = raw.get("max_entries", 4)
        if type(enabled) is not bool:
            raise TypeError("additional_config.ltx2_audio_cuda_graph.enabled must be a bool")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("additional_config.ltx2_audio_cuda_graph.max_entries must be a positive integer")
        return cls(enabled=enabled, max_entries=max_entries)


@dataclass(frozen=True)
class LTX2AudioGraphKey:
    """Structural fields that determine buffers and Python control flow."""

    hidden_shape: tuple[int, ...]
    context_shape: tuple[int, ...]
    has_audio_attention_mask: bool
    has_perturbation_mask: bool
    stg_blocks: tuple[int, ...] | None


@dataclass
class LTX2AudioGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_hidden_states: torch.Tensor
    static_context: torch.Tensor
    static_timestep: torch.Tensor
    static_sigma: torch.Tensor
    static_coords: torch.Tensor
    static_attention_mask: torch.Tensor | None
    static_perturbation_mask: torch.Tensor | None
    static_output: torch.Tensor


def make_ltx2_audio_graph_key(
    hidden_states: torch.Tensor,
    context: torch.Tensor,
    audio_attention_mask: torch.Tensor | None,
    perturbation_mask: torch.Tensor | None,
    stg_blocks: Sequence[int] | None,
) -> LTX2AudioGraphKey:
    blocks = None if stg_blocks is None else tuple(sorted(set(int(block) for block in stg_blocks)))
    return LTX2AudioGraphKey(
        hidden_shape=tuple(hidden_states.shape),
        context_shape=tuple(context.shape),
        has_audio_attention_mask=audio_attention_mask is not None,
        has_perturbation_mask=perturbation_mask is not None,
        stg_blocks=blocks,
    )


def _static_copy(value: torch.Tensor) -> torch.Tensor:
    static = torch.empty_like(value, memory_format=torch.contiguous_format)
    static.copy_(value)
    return static


def _has_tensor_metadata(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> bool:
    """Check CUDA Graph input metadata without constraining source strides."""
    return value.device == device and value.dtype == dtype and (shape is None or tuple(value.shape) == shape)


def _same_tensor_metadata(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.layout == right.layout
    )


class LTX2AudioCUDAGraphRunner:
    """Capture and replay complete LTX2 audio Transformer forwards.

    A runner is owned by one diffusion worker and is accessed serially by that
    worker. Captures and replays use a runner-private shared graph pool.
    Returned outputs are cloned so callers never retain graph storage.
    Concurrent calls are unsupported because entries reuse mutable static
    buffers across the complete copy, replay, and output-clone lifecycle.
    """

    def __init__(
        self,
        transformer: Any,
        *,
        max_graphs: int = 4,
        device: torch.device | str | None = None,
    ) -> None:
        if type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        self.transformer = transformer
        self.max_graphs = max_graphs
        if device is None:
            try:
                device = next(transformer.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self._cache: OrderedDict[LTX2AudioGraphKey, LTX2AudioGraphEntry] = OrderedDict()
        self._failed_keys: OrderedDict[LTX2AudioGraphKey, None] = OrderedDict()
        self._pool: Any | None = None
        self._stats = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "capture_failures": 0,
            "evictions": 0,
            "eager": 0,
        }
        self.last_call_info: dict[str, Any] = {}

    def stats_snapshot(self) -> dict[str, int]:
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "failed_key_count": len(self._failed_keys),
            "max_graphs": self.max_graphs,
        }

    @staticmethod
    def _canonical_blocks(stg_blocks: Sequence[int] | None) -> tuple[int, ...] | None:
        return None if stg_blocks is None else tuple(sorted(set(int(block) for block in stg_blocks)))

    def _inputs_are_compatible(
        self,
        *,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
    ) -> bool:
        expected_device = self.device
        if expected_device.type != "cuda" or not torch.cuda.is_available():
            return False

        if hidden_states.ndim != 3 or not _has_tensor_metadata(
            hidden_states,
            device=expected_device,
            dtype=torch.bfloat16,
        ):
            return False

        batch, audio_tokens, _ = hidden_states.shape
        if context.ndim != 3 or context.shape[0] != batch:
            return False

        required_inputs = (
            (context, torch.bfloat16, None),
            (timestep, torch.float32, (batch, audio_tokens)),
            (sigma, torch.float32, (batch,)),
            (coords, torch.float32, (batch, 1, audio_tokens, 2)),
        )
        if not all(
            _has_tensor_metadata(value, device=expected_device, dtype=dtype, shape=shape)
            for value, dtype, shape in required_inputs
        ):
            return False

        if attention_mask is not None and not _has_tensor_metadata(
            attention_mask,
            device=expected_device,
            dtype=torch.bool,
            shape=(batch, audio_tokens),
        ):
            return False

        return perturbation_mask is None or _has_tensor_metadata(
            perturbation_mask,
            device=expected_device,
            dtype=torch.bfloat16,
            shape=(batch, 1, 1),
        )

    @staticmethod
    def _entry_matches(
        entry: LTX2AudioGraphEntry,
        *,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
    ) -> bool:
        pairs = (
            (entry.static_hidden_states, hidden_states),
            (entry.static_context, context),
            (entry.static_timestep, timestep),
            (entry.static_sigma, sigma),
            (entry.static_coords, coords),
            (entry.static_attention_mask, attention_mask),
            (entry.static_perturbation_mask, perturbation_mask),
        )
        for static, current in pairs:
            if static is None or current is None:
                if static is not current:
                    return False
                continue
            if not _same_tensor_metadata(static, current):
                return False
        return True

    def _call_transformer(
        self,
        *,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
        stg_blocks: tuple[int, ...] | None,
    ) -> torch.Tensor:
        attention_kwargs: dict[str, Any] = {}
        if perturbation_mask is not None:
            attention_kwargs["ltx_perturbation_kwargs"] = {
                "audio_self_attention_mask": perturbation_mask,
                "audio_self_attention_blocks": stg_blocks,
            }
        return self.transformer(
            audio_hidden_states=hidden_states,
            audio_encoder_hidden_states=context,
            audio_timestep=timestep,
            audio_sigma=sigma,
            audio_coords=coords,
            audio_attention_mask=attention_mask,
            attention_kwargs=attention_kwargs,
        )

    def _eager_forward(self, **kwargs: Any) -> torch.Tensor:
        self._stats["eager"] += 1
        return self._call_transformer(**kwargs)

    def _capture(self, **inputs: Any) -> LTX2AudioGraphEntry:
        static_inputs = {
            name: _static_copy(value) if isinstance(value, torch.Tensor) else value for name, value in inputs.items()
        }
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                self._call_transformer(**static_inputs)
        current_stream.wait_stream(warmup_stream)
        torch.accelerator.synchronize(self.device)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with (
            torch.no_grad(),
            torch.cuda.graph(
                graph,
                pool=self._pool,
                # This controls capture-time error detection only; it does not
                # bind the resulting graph to the capturing Python thread.
                capture_error_mode="thread_local",
            ),
        ):
            static_output = self._call_transformer(**static_inputs)
        return LTX2AudioGraphEntry(
            graph=graph,
            static_hidden_states=static_inputs["hidden_states"],
            static_context=static_inputs["context"],
            static_timestep=static_inputs["timestep"],
            static_sigma=static_inputs["sigma"],
            static_coords=static_inputs["coords"],
            static_attention_mask=static_inputs["attention_mask"],
            static_perturbation_mask=static_inputs["perturbation_mask"],
            static_output=static_output,
        )

    @staticmethod
    def _copy_and_replay(
        entry: LTX2AudioGraphEntry,
        *,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
        stg_blocks: tuple[int, ...] | None,
    ) -> torch.Tensor:
        del stg_blocks
        entry.static_hidden_states.copy_(hidden_states)
        entry.static_context.copy_(context)
        entry.static_timestep.copy_(timestep)
        entry.static_sigma.copy_(sigma)
        entry.static_coords.copy_(coords)
        if entry.static_attention_mask is not None:
            assert attention_mask is not None
            entry.static_attention_mask.copy_(attention_mask)
        if entry.static_perturbation_mask is not None:
            assert perturbation_mask is not None
            entry.static_perturbation_mask.copy_(perturbation_mask)
        entry.graph.replay()
        return entry.static_output.detach().clone()

    def __call__(
        self,
        *,
        audio_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        audio_timestep: torch.Tensor,
        audio_sigma: torch.Tensor,
        audio_coords: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        stg_blocks: Sequence[int] | None = None,
    ) -> torch.Tensor:
        self._stats["calls"] += 1
        blocks = self._canonical_blocks(stg_blocks)
        inputs = {
            "hidden_states": audio_hidden_states,
            "context": audio_encoder_hidden_states,
            "timestep": audio_timestep,
            "sigma": audio_sigma,
            "coords": audio_coords,
            "attention_mask": audio_attention_mask,
            "perturbation_mask": perturbation_mask,
            "stg_blocks": blocks,
        }
        if not self._inputs_are_compatible(**{k: v for k, v in inputs.items() if k != "stg_blocks"}):
            self.last_call_info = {"mode": "eager", "reason": "incompatible_inputs"}
            logger.debug("LTX2 audio CUDA Graph bypassed for incompatible request inputs")
            return self._eager_forward(**inputs)

        if torch.cuda.is_current_stream_capturing():
            self.last_call_info = {"mode": "eager", "reason": "active_capture"}
            return self._eager_forward(**inputs)

        key = make_ltx2_audio_graph_key(
            audio_hidden_states,
            audio_encoder_hidden_states,
            audio_attention_mask,
            perturbation_mask,
            blocks,
        )
        entry = self._cache.get(key)
        if entry is not None:
            if not self._entry_matches(entry, **{k: v for k, v in inputs.items() if k != "stg_blocks"}):
                self.last_call_info = {"mode": "eager", "reason": "incompatible_inputs", "key": key}
                return self._eager_forward(**inputs)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            self.last_call_info = {"mode": "replay", "reason": "cache_hit", "key": key}
            return self._copy_and_replay(entry, **inputs)

        if key in self._failed_keys:
            self._failed_keys.move_to_end(key)
            self.last_call_info = {"mode": "eager", "reason": "previous_capture_failure", "key": key}
            return self._eager_forward(**inputs)

        try:
            entry = self._capture(**inputs)
            self._stats["captures"] += 1
            self._cache[key] = entry
            while len(self._cache) > self.max_graphs:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                self._stats["evictions"] += 1
            self.last_call_info = {"mode": "capture", "reason": "cache_miss", "key": key}
            return self._copy_and_replay(entry, **inputs)
        except Exception:
            self._stats["capture_failures"] += 1
            self._failed_keys[key] = None
            self._failed_keys.move_to_end(key)
            while len(self._failed_keys) > self.max_graphs:
                self._failed_keys.popitem(last=False)
            self.last_call_info = {"mode": "eager", "reason": "capture_failure", "key": key}
            logger.warning(
                "LTX2 audio CUDA Graph capture failed; using eager execution for this signature",
                exc_info=True,
            )
            return self._eager_forward(**inputs)

    def clear(self) -> None:
        """Synchronously release captured graphs and reset the lifecycle."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.accelerator.synchronize(self.device)
        self._cache.clear()
        self._failed_keys.clear()
        self._pool = None
        for name in self._stats:
            self._stats[name] = 0
        self.last_call_info = {}
