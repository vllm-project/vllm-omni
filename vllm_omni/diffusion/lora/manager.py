# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Lightweight LoRA manager for diffusion pipelines.

Design goals:
- Request-driven lazy loading with per-worker LRU cache
- Path whitelist to avoid arbitrary filesystem access
- Graceful no-op if a LoRA is missing or incompatible

Current implementation focuses on torch.nn.Linear targets (text encoders, etc.).
Custom kernels (e.g., QKVParallelLinear) are left untouched to avoid instability;
this keeps base quality while enabling common SD/SD3 LoRAs that only touch
standard Linear layers in text encoders / VAE.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from safetensors.torch import load_file
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


@dataclass
class _PatchedModule:
    module: torch.nn.Module
    orig_forward: torch.nn.Module
    lora_a: torch.nn.Parameter
    lora_b: torch.nn.Parameter
    scaling: float


@dataclass
class _AdapterHandle:
    name: str
    path: str
    scale: float
    patched: list[_PatchedModule]
    last_used: float


def _get_attr(obj, names: Iterable[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


class DiffusionLoRAManager:
    """Per-worker LoRA cache and injector."""

    def __init__(
        self,
        pipeline: torch.nn.Module,
        device: torch.device,
        *,
        dtype: torch.dtype,
        max_cache_size: int = 4,
        allowed_dirs: list[str] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else []
        self.max_cache_size = max(1, int(max_cache_size))
        self.cache: OrderedDict[str, _AdapterHandle] = OrderedDict()

    # Public API -----------------------------------------------------
    def set_active_adapter(self, lora_req: LoRARequest | None) -> None:
        if lora_req is None:
            return
        name, path, scale = self._normalize_request(lora_req)
        if path is None:
            logger.warning("LoRA request missing path; skip")
            return
        handle = self._ensure_loaded(name, path, scale)
        if handle is None:
            return
        self._activate(handle)

    # Internal helpers ----------------------------------------------
    def _normalize_request(self, lora_req: LoRARequest) -> tuple[str, str | None, float]:
        name = _get_attr(lora_req, ["name", "lora_name", "adapter_name", "lora_nickname"], None)
        int_id = _get_attr(lora_req, ["lora_int_id", "int_id"], None)
        if name is None and int_id is not None:
            name = str(int_id)
        path = _get_attr(lora_req, ["local_path", "lora_local_path", "lora_path", "path"], None)
        scale = float(_get_attr(lora_req, ["scale", "lora_scale"], 1.0))

        if path is not None:
            path = os.path.realpath(path)
            if self.allowed_dirs:
                if not any(path.startswith(root + os.sep) or path == root for root in self.allowed_dirs):
                    logger.warning("LoRA path %s not in whitelist %s", path, self.allowed_dirs)
                    return name or "unnamed", None, scale
        return name or "unnamed", path, scale

    def _ensure_loaded(self, name: str, path: str, scale: float) -> _AdapterHandle | None:
        # Hit
        if name in self.cache:
            handle = self.cache.pop(name)
            self.cache[name] = handle
            handle.last_used = time.time()
            # Update scale per request
            handle.scale = scale
            logger.info("LoRA hit: %s", name)
            return handle

        # Load
        try:
            handle = self._load_adapter(name, path, scale)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load LoRA %s from %s: %s", name, path, exc)
            return None

        self.cache[name] = handle
        self._evict_if_needed()
        return handle

    def _load_adapter(self, name: str, path: str, scale: float) -> _AdapterHandle:
        logger.info("Loading LoRA %s from %s", name, path)
        state_dict = load_file(path, device="cpu")
        keys = list(state_dict.keys())
        # Group by module prefix ending with .lora_down / .lora_up
        pairs = {}
        for k in keys:
            if k.endswith("lora_down.weight"):
                base = k[: -len("lora_down.weight")]
                pairs.setdefault(base, {})["down"] = state_dict[k]
            elif k.endswith("lora_up.weight"):
                base = k[: -len("lora_up.weight")]
                pairs.setdefault(base, {})["up"] = state_dict[k]
            elif k.endswith("alpha"):
                base = k[: -len("alpha")]
                pairs.setdefault(base, {})["alpha"] = state_dict[k]

        patched: list[_PatchedModule] = []
        modules = dict(self.pipeline.named_modules())

        for base, parts in pairs.items():
            if "down" not in parts or "up" not in parts:
                continue
            mod = modules.get(base.rstrip("."))
            if mod is None or not isinstance(mod, torch.nn.Linear):
                continue

            down = parts["down"].to(device=self.device, dtype=self.dtype)
            up = parts["up"].to(device=self.device, dtype=self.dtype)
            rank = down.shape[0]
            alpha = parts.get("alpha")
            alpha_val = float(alpha.item()) if alpha is not None else rank
            base_scaling = alpha_val / max(rank, 1)
            scaling = scale * base_scaling

            self._apply_lora_to_linear(mod, down, up, scaling, base_scaling, patched)

        handle = _AdapterHandle(
            name=name,
            path=path,
            scale=scale,
            patched=patched,
            last_used=time.time(),
        )
        logger.info("LoRA %s loaded; patched %d modules", name, len(patched))
        return handle

    def _apply_lora_to_linear(
        self,
        module: torch.nn.Linear,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        scaling: float,
        base_scaling: float,
        patched: list[_PatchedModule],
    ) -> None:
        if hasattr(module, "_omni_lora_patched"):
            # Update existing LoRA weights
            module._omni_lora_down.data.copy_(lora_down)
            module._omni_lora_up.data.copy_(lora_up)
            module._omni_lora_scale = scaling
            return

        module._omni_lora_down = torch.nn.Parameter(lora_down)
        module._omni_lora_up = torch.nn.Parameter(lora_up)
        module._omni_lora_scale = scaling
        module._omni_lora_base = base_scaling
        module._omni_lora_patched = True

        orig_forward = module.forward

        def lora_forward(x, *, orig_forward=orig_forward, mod=module):
            base = orig_forward(x)
            # x: (..., in_features)
            lora_out = (x @ mod._omni_lora_down.t()) @ mod._omni_lora_up.t()
            return base + lora_out * mod._omni_lora_scale

        module.forward = lora_forward  # type: ignore[assignment]
        patched.append(
            _PatchedModule(
                module=module,
                orig_forward=orig_forward,
                lora_a=module._omni_lora_down,
                lora_b=module._omni_lora_up,
                scaling=base_scaling,
            )
        )

    def _activate(self, handle: _AdapterHandle) -> None:
        # Activation is already applied; we only refresh scale values.
        for entry in handle.patched:
            entry.module._omni_lora_scale = handle.scale * entry.scaling
        handle.last_used = time.time()

    def _evict_if_needed(self) -> None:
        while len(self.cache) > self.max_cache_size and len(self.cache) > 0:
            name, handle = self.cache.popitem(last=False)
            self._unload(handle)
            logger.info("Evicted LoRA %s to keep cache size <= %d", name, self.max_cache_size)

    def _unload(self, handle: _AdapterHandle) -> None:
        for entry in handle.patched:
            entry.module.forward = entry.orig_forward  # type: ignore[assignment]
            for attr in [
                "_omni_lora_patched",
                "_omni_lora_down",
                "_omni_lora_up",
                "_omni_lora_scale",
                "_omni_lora_base",
            ]:
                if hasattr(entry.module, attr):
                    delattr(entry.module, attr)
        torch.cuda.empty_cache()
