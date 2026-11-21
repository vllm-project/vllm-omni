# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fuse one H3 adapter into floating checkpoint tensors before TP and quantization."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.lora.request import LoRARequest


@contextmanager
def _npu_fp32_matmul() -> Iterator[None]:
    """Keep startup fusion in FP32 without changing the serving precision policy."""
    import torch_npu

    matmul = torch_npu.npu.matmul
    previous_hf32 = matmul.allow_hf32
    previous_cube = getattr(matmul, "cube_math_type", None)
    try:
        matmul.allow_hf32 = False
        # An unset cube policy follows allow_hf32. Leave it unset so its
        # original default remains intact after fusion.
        if previous_cube is not None:
            matmul.cube_math_type = type(previous_cube).KEEP_DTYPE
        yield
    finally:
        if previous_cube is not None:
            matmul.cube_math_type = previous_cube
        matmul.allow_hf32 = previous_hf32


@dataclass(frozen=True)
class _Factors:
    a: torch.Tensor
    b: torch.Tensor
    strength: float


class MiniMaxH3StartupLoRA:
    """Merge canonical global H3 weights before a loader can shard or quantize them."""

    _ROWS_PER_TILE = 512
    _NPU_ROWS_PER_TILE = 4096

    def __init__(self, model: LoRAModel, scale: float) -> None:
        if not math.isfinite(scale):
            raise ValueError("startup-merge requires a finite LoRA scale")
        self._factors: dict[tuple[str, str | int | None], _Factors] = {}
        self._consumed: set[tuple[str, str | int | None]] = set()
        for raw_name, weights in model.loras.items():
            name = raw_name.removeprefix("transformer.")
            projection = name.rsplit(".", 1)[-1]
            if projection in {"to_q", "to_k", "to_v"}:
                self._add(
                    name.rsplit(".", 1)[0] + ".qkv_proj",
                    projection[-1],
                    weights.lora_a,
                    weights.lora_b,
                    weights.scaling * scale,
                )
            elif isinstance(weights, PackedLoRALayerWeights):
                labels: tuple[str | int, ...]
                if name.endswith(".attn.qkv_proj"):
                    labels = ("q", "k", "v")
                elif name.endswith(".mlp.fc1"):
                    labels = (0, 1)
                else:
                    raise ValueError(f"Unsupported startup-merge packed target: {raw_name}")
                for label, a, b, strength in zip(labels, weights.lora_a, weights.lora_b, weights.scaling, strict=True):
                    if a is not None or b is not None:
                        self._add(name, label, a, b, strength * scale)
            else:
                self._add(name, None, weights.lora_a, weights.lora_b, weights.scaling * scale)
        if not self._factors:
            raise ValueError("startup-merge adapter has no LoRA factors")
        for name, shard in self._factors:
            if shard is not None and (name, None) in self._factors:
                raise ValueError(f"Conflicting fused and split startup-merge targets: {name}")

    def _add(
        self, name: str, shard: str | int | None, a: torch.Tensor | None, b: torch.Tensor | None, strength: float
    ) -> None:
        if a is None or b is None or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid startup-merge LoRA factor shapes for {name}")
        if not math.isfinite(strength) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Non-finite startup-merge LoRA factors for {name}")
        key = (name, shard)
        if key in self._factors:
            raise ValueError(f"Duplicate startup-merge target: {key}")
        self._factors[key] = _Factors(a.float().cpu(), b.float().cpu(), strength)

    @torch.no_grad()
    def merge(
        self,
        name: str,
        weight: torch.Tensor,
        shard: str | int | None = None,
        *,
        output_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Merge a canonical checkpoint matrix, or one packed projection."""
        if not name.endswith(".weight"):
            return weight
        target = name.removesuffix(".weight")
        device = weight.device if device is None else torch.device(device)
        merge_device = device if device.type == "npu" else torch.device("cpu")
        key = (target, shard)
        factors = self._factors.get(key)
        if factors is None and shard is None and target.endswith(".attn.qkv_proj"):
            labels = ("q", "k", "v")
            if not any((target, label) in self._factors for label in labels):
                return weight
            if weight.shape[0] % len(labels):
                raise ValueError(f"Invalid startup-merge packed checkpoint shape: {name}")
            return torch.cat(
                [
                    self.merge(name, part, label, output_dtype=output_dtype, device=merge_device).to(
                        device=merge_device, dtype=output_dtype or weight.dtype
                    )
                    for part, label in zip(weight.chunk(len(labels)), labels)
                ]
            )
        if factors is None and shard is not None:
            full = self._factors.get((target, None))
            if full is None:
                return weight
            index = ("q", "k", "v").index(shard) if isinstance(shard, str) else shard
            rows = weight.shape[0]
            count = 3 if isinstance(shard, str) else 2
            if full.b.shape[0] != rows * count:
                raise ValueError(f"Invalid startup-merge fused factor shape: {name}")
            factors = _Factors(full.a, full.b[index * rows : (index + 1) * rows], full.strength)
        if factors is None:
            return weight
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("startup-merge requires original FP16/BF16/FP32 checkpoint weights")
        if weight.ndim != 2 or weight.shape != (factors.b.shape[0], factors.a.shape[1]):
            raise ValueError(f"startup-merge LoRA factor shapes do not match checkpoint: {name}")
        if key in self._consumed:
            raise ValueError(f"Duplicate startup-merge checkpoint target: {key}")
        dtype = weight.dtype if output_dtype is None else output_dtype
        result = self._merge_tiles(weight, factors, dtype, merge_device, name)
        self._consumed.add(key)
        return result

    def _merge_tiles(
        self, source: torch.Tensor, factors: _Factors, dtype: torch.dtype, device: torch.device, name: str
    ) -> torch.Tensor:
        """Apply the same FP32 tile calculation with backend-specific staging."""
        on_npu = device.type == "npu"
        a, b = factors.a, factors.b
        if on_npu:
            # Upload once and keep the merged tensor on the device for TP and quantization.
            source, a, b = source.to(device), a.to(device), b.to(device)
        rows = self._NPU_ROWS_PER_TILE if on_npu else self._ROWS_PER_TILE
        result = torch.empty_like(source, dtype=dtype, device=device)
        npu_checks: list[torch.Tensor] = []
        error = f"startup-merge produced non-finite weights: {name}"
        with _npu_fp32_matmul() if on_npu else nullcontext():
            for start in range(0, source.shape[0], rows):
                end = min(start + rows, source.shape[0])
                base = source[start:end].float()
                if not on_npu:
                    base = base.cpu()
                value = torch.addmm(base, b[start:end], a, alpha=factors.strength).to(dtype)
                finite = torch.isfinite(value).all()
                if on_npu:
                    npu_checks.append(finite)
                elif not finite:
                    raise ValueError(error)
                result[start:end].copy_(value)
            # Do not synchronize the NPU between tiles.
            if on_npu and not torch.stack(npu_checks).all().item():
                raise ValueError(error)
        return result

    def load_weight(
        self,
        name: str,
        loader: Callable[..., None],
        param: torch.Tensor,
        tensor: torch.Tensor,
        shard_id: str | int | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        """Fuse after layout conversion, before the original TP/quantizing loader."""
        # Preserve FP32 target layers even when the source
        # checkpoint is BF16. Do not round their fused delta back to BF16.
        dtype = param.dtype if param.dtype in (torch.float16, torch.bfloat16, torch.float32) else None
        merged = self.merge(
            name, tensor, shard_id, output_dtype=dtype, device=param.device if device is None else device
        )
        if shard_id is None:
            loader(param, merged)
        else:
            loader(param, merged, shard_id)

    def validate(self) -> None:
        """Reject adapters with targets missing from the original checkpoint."""
        missing = []
        for name, shard in self._factors:
            if (name, shard) in self._consumed:
                continue
            labels = ("q", "k", "v") if name.endswith(".attn.qkv_proj") else (0, 1)
            if shard is None and all((name, label) in self._consumed for label in labels):
                continue
            missing.append((name, shard))
        if missing:
            raise ValueError(f"startup-merge adapter targets were not loaded: {missing[:8]}")


def load_startup_peft_adapter(path: str, request: LoRARequest) -> LoRAModel:
    """Load ordinary PEFT when H3's format-specific loaders decline the path."""
    helper = PEFTHelper.from_local_dir(path, max_position_embeddings=None)
    if helper.modules_to_save:
        raise ValueError("startup-merge supports A/B adapters without modules_to_save")
    # Validate the PEFT transformation without imposing the runtime kernel's rank limit.
    unsupported = helper._validate_features()
    if helper.bias != "none":
        unsupported.append("Adapter bias is not supported.")
    if unsupported:
        raise ValueError(f"startup-merge cannot load this PEFT adapter: {' '.join(unsupported)}")
    return LoRAModel.from_local_checkpoint(
        path,
        expected_lora_modules={"qkv_proj", "to_q", "to_k", "to_v", "out_proj", "fc1", "fc2", "linear"},
        peft_helper=helper,
        lora_model_id=request.lora_int_id,
        device="cpu",
        dtype=torch.float32,
        model_vocab_size=None,
        weights_mapper=None,
    )
