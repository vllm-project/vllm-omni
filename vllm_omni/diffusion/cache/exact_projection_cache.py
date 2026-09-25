# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent exact caching of small conditioning projections."""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn.modules import module as module_hooks


class ExactProjectionCache:
    """Bounded, exact memoization for deterministic conditioning projections.

    Models call :meth:`prepare` once for an immutable conditioning tensor, then
    :meth:`project` for each leaf projection using a stable, model-defined name.
    The supplied computation must depend only on that tensor, the projection's
    parameters/buffers and fixed preprocessing. Include all request conditioning
    in the tensor; timestep alone is insufficient for prompt/guidance-dependent
    projections. Outputs returned on cache hits must be treated as read-only.

    The original projection computes misses, preserving its quantization, LoRA
    and TP behavior. Parameter/buffer versions guard reuse, and TP ranks vote
    before skipping collectives. Call :meth:`clear` for adapter/configuration
    changes or model moves. Gradient-enabled and compiled execution bypass reuse.

    Each cache belongs to one model instance executing forwards serially. It
    retains a bounded subset across requests, so it is independent of the
    request-scoped approximate cache backends. It does not offload weights or
    suppress block prefetch. Models may supply validated precomputed results by
    overriding :meth:`_lookup_precomputed`; artifact loading stays model-owned.
    """

    def __init__(self, *, max_bytes: int = 256 * 1024**2) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Projection cache budget must be a nonnegative byte count")
        self.max_bytes = max_bytes
        self._entries: dict[tuple[str, str, tuple[Any, ...]], tuple[Any, torch.Tensor, int]] = {}
        self._bytes = 0
        self._input: weakref.ReferenceType[torch.Tensor] | None = None
        self._key: str | None = None
        self.hits = self.misses = 0

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
        self._input = None
        self._key = None

    def prepare(self, embedding: torch.Tensor) -> None:
        self._input = None
        self._key = None
        if not self.max_bytes or torch.is_grad_enabled() or torch.compiler.is_compiling():
            return
        self._key = tensor_digest(embedding)
        self._input = weakref.ref(embedding)

    @staticmethod
    def _numerical_settings(device_type: str) -> tuple[Any, ...]:
        # A projection may enter its own autocast context after prepare().
        return (
            torch.get_float32_matmul_precision(),
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        )

    @staticmethod
    def _has_forward_hooks(module: nn.Module) -> bool:
        # Match nn.Module's global hooks as well as hooks on wrapped linears.
        return bool(
            module_hooks._global_forward_pre_hooks
            or module_hooks._global_forward_hooks
            or any(child._forward_pre_hooks or child._forward_hooks for child in module.modules())
        )

    @staticmethod
    def _signature(module: nn.Module) -> tuple[Any, ...]:
        # A replaced or restored storage conservatively causes a miss. The
        # containing block's offload hooks still run before this method.
        tensors = [*module.named_parameters(), *module.named_buffers()]
        # vLLM LoRA stores these tensors outside registered buffers. Its
        # suspend/resume mask can change without changing the base weights.
        for field in ("lora_a_stacked", "lora_b_stacked"):
            tensors.extend((f"{field}.{i}", value) for i, value in enumerate(getattr(module, field, ())))
        return (
            id(module),
            id(getattr(module, "quant_method", None)),
            getattr(module, "_diffusion_lora_active_slices", None),
            *(
                (name, id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
                for name, value in tensors
            ),
        )

    def project(
        self,
        name: str,
        module: nn.Module,
        embedding: torch.Tensor,
        compute: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if (
            torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or self._key is None
            or self._input is None
            or self._input() is not embedding
        ):
            return compute()
        key = self._key, name, self._numerical_settings(embedding.device.type)
        signature: tuple[Any, ...] | None
        try:
            signature = self._signature(module)
            if self._has_forward_hooks(module):
                # A linear hook may mutate weights or transform the output;
                # executing it only on misses would change its semantics.
                signature = None
        except RuntimeError:
            # Inference tensors without version counters cannot establish that
            # weights stayed unchanged. Still participate in the TP vote.
            signature = None
        cached = self._entries.get(key)
        hit = signature is not None and cached is not None and cached[0] == signature
        from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group

        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            vote = torch.tensor([int(hit)], device=embedding.device, dtype=torch.int32)
            hit = int(get_tp_group().all_reduce(vote).item()) == world_size
        if hit:
            assert cached is not None
            self.hits += 1
            return cached[1]
        self.misses += 1
        value = (
            self._lookup_precomputed(name, embedding, signature) if world_size == 1 and signature is not None else None
        )
        if value is None:
            value = compute()
        if cached is not None:
            self._bytes -= self._entries.pop(key)[2]
        size = value.numel() * value.element_size()
        if signature is not None and self._bytes + size <= self.max_bytes:
            # Retain a reusable subset when a long denoising schedule exceeds
            # the budget. LRU would evict every entry before the next request
            # reaches it, producing zero hits for a repeated cyclic schedule.
            # Own the cached storage even if a provider returns workspace views.
            self._entries[key] = (signature, value.detach().clone(), size)
            self._bytes += size
        return value

    def _lookup_precomputed(self, name: str, embedding: torch.Tensor, signature: Any) -> torch.Tensor | None:
        """Return a validated TP1 projection, or None to execute the original.

        Called only on runtime-cache misses with a usable weight signature.
        Implementations must also validate inputs, numerical settings, artifact
        compatibility and any state not represented in the module signature.
        """
        return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash dtype, shape and actual bytes, including BF16, without widening."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(canonical_json([str(value.dtype), list(value.shape)]).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()
