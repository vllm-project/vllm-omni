# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict, streaming readers for offline ModelSlim H3 component checkpoints.

MX reconstruction either provides explicit floating-point compatibility or
feeds the Ascend native MX adapter, which encodes weights once at load time.
The reader itself performs no activation quantization or GEMM.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

_MX8 = "W8A8_MXFP8"
_MX4_DUAL = "W4A4_MXFP4_DUALSCALE"
_MX4_SINGLE = "W4A4_MXFP4"
_MX4_SCHEMES = {_MX4_SINGLE, _MX4_DUAL}
_MX_SCHEMES = {*_MX4_SCHEMES, _MX8}
_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


def npu_quant_family(device_name: str) -> str:
    """Explicit device allowlist; unknown NPU generations never guess a format."""
    name = device_name.upper().replace("_", "").replace("-", "")
    if name.startswith("ASCEND950"):
        return "mx"
    raise ValueError(f"unsupported H3 native MX NPU: {device_name}")


@lru_cache(maxsize=16)
def _device_quant_family(index: int) -> str:
    import torch_npu

    return npu_quant_family(torch_npu.npu.get_device_name(index))


def resolve_npu_quant_family(device: str | None = None) -> str:
    import torch_npu

    target = torch.device(device) if device is not None else None
    index = target.index if target is not None else None
    if index is None:
        index = torch_npu.npu.current_device()
    return _device_quant_family(index)


def require_native_family(family: str, device: torch.device) -> None:
    if device.type != "npu" or resolve_npu_quant_family(str(device)) != family:
        raise ValueError(f"H3 native {family} weights are incompatible with the selected NPU")


def unpack_fp4(weight: torch.Tensor) -> torch.Tensor:
    """Decode ModelSlim E2M1: the first K element occupies the low nibble."""
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("packed FP4 weight must be a uint8 matrix")
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=weight.device)
    return values[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)


def reconstruct_mx_weight(
    weight: torch.Tensor, scale: torch.Tensor, dual_scale: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct weights without requantization, retaining both scale levels."""
    if scale.dtype != torch.uint8 or scale.ndim != 2 or torch.any(scale == 255):
        raise ValueError("MX scale must be a finite E8M0 uint8 matrix")
    numeric = unpack_fp4(weight) if weight.dtype == torch.uint8 else weight.float()
    if numeric.ndim != 2 or scale.shape != (numeric.shape[0], (numeric.shape[1] + 31) // 32):
        raise ValueError("MX scale shape must be [N, ceil(K/32)]")
    k = numeric.shape[1]
    if k % 32:
        # The delivered VAE includes K=24: its last group uses one scale
        # for the remaining elements, without padding the stored weight.
        numeric = torch.nn.functional.pad(numeric, (0, 32 - k % 32))
    exponent = scale.to(torch.int32) - 127
    # ldexp also handles E8M0=0 (2**-127), without a host float exponent cast.
    blocks = torch.ldexp(numeric.reshape(*scale.shape, 32), exponent.unsqueeze(-1))
    if dual_scale is not None:
        if dual_scale.ndim != 3 or dual_scale.shape[0] != scale.shape[0] or dual_scale.shape[-1] != 1:
            raise ValueError("dual scale must have shape [N, K/dual_block_size, 1]")
        if dual_scale.shape[1] == 0 or scale.shape[1] % dual_scale.shape[1]:
            raise ValueError("dual scale groups must divide the fine scale groups")
        if not torch.isfinite(dual_scale).all() or torch.any(dual_scale <= 0):
            raise ValueError("dual scale must be finite and positive")
        coarse = dual_scale.repeat_interleave(scale.shape[1] // dual_scale.shape[1], dim=1)
        blocks = blocks * coarse
    return blocks.flatten(1)[:, :k]


@dataclass(frozen=True)
class TensorSpec:
    file: Path
    dtype: str
    shape: tuple[int, ...]


class OfflineComponentCheckpoint:
    """Indexed CPU reader; metadata is checked before tensor materialization."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.execution_mode = "dequantize"
        self.single_scale_mx4 = False
        if not self.path.is_dir():
            raise ValueError(f"offline component directory does not exist: {self.path}")
        description_path = self.path / "quant_model_description.json"
        self.description = json.loads(description_path.read_text(encoding="utf-8"))
        indices = [
            self.path / name
            for name in (
                "model.safetensors.index.json",
                "diffusion_pytorch_model.safetensors.index.json",
                "quant_model_weights.safetensors.index.json",
            )
            if (self.path / name).is_file()
        ]
        index = json.loads(indices[0].read_text(encoding="utf-8"))["weight_map"] if indices else None
        filenames = (
            sorted(set(index.values()))
            if index is not None
            else sorted(p.name for p in self.path.glob("*.safetensors"))
        )
        if not filenames:
            raise ValueError(f"no safetensors weights in {self.path}")
        self.tensors: dict[str, TensorSpec] = {}
        for filename in filenames:
            file = (self.path / filename).resolve()
            if not file.is_relative_to(self.path) or not file.is_file():
                raise ValueError(f"invalid or missing offline weight shard: {filename}")
            with safe_open(str(file), framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if name in self.tensors:
                        raise ValueError(f"duplicate offline tensor: {name}")
                    if index is not None and index.get(name) != filename:
                        raise ValueError(f"offline index disagrees with shard for {name}")
                    view = reader.get_slice(name)
                    self.tensors[name] = TensorSpec(file, view.get_dtype(), tuple(view.get_shape()))
        if index is not None and set(index) != self.tensors.keys():
            raise ValueError("offline index references missing tensors")
        records = {
            name: scheme
            for name, scheme in self.description.items()
            if name.endswith((".weight", ".bias", ".weight_scale", ".weight_dual_scale"))
        }
        if missing := records.keys() - self.tensors.keys():
            raise ValueError(f"offline description references missing tensors: {sorted(missing)[:8]}")
        for name, spec in self.tensors.items():
            scheme = self.description.get(name, "FLOAT")
            if scheme not in {"FLOAT", *_MX_SCHEMES}:
                raise ValueError(f"unsupported offline MX scheme {scheme!r} for {name}")
            if scheme == "FLOAT":
                if spec.dtype not in _FLOAT_DTYPES:
                    raise ValueError(f"quantized tensor {name} lacks a supported scheme")
                continue
            module, _, kind = name.rpartition(".")
            if kind == "bias" and spec.dtype in _FLOAT_DTYPES:
                # ModelSlim can tag a floating bias with its linear's scheme.
                continue
            if kind not in {"weight", "weight_scale", "weight_dual_scale"}:
                raise ValueError(f"unsupported quantized tensor: {name}")
            if kind != "weight":
                if self.description.get(module + ".weight") != scheme:
                    raise ValueError(f"orphan or inconsistent quantization scale: {name}")
                continue
            expected_dtypes = {
                _MX8: {"F8_E4M3"},
                _MX4_SINGLE: {"U8"},
                _MX4_DUAL: {"U8"},
            }[scheme]
            if spec.dtype not in expected_dtypes or len(spec.shape) != 2:
                raise ValueError(f"{name}: {scheme} requires {expected_dtypes} matrix, got {spec}")
            scale = self.tensors.get(module + ".weight_scale")
            n, packed_k = spec.shape
            k = packed_k * (2 if scheme in _MX4_SCHEMES else 1)
            groups = (k + 31) // 32
            if scale is None or scale.shape != (n, groups):
                raise ValueError(f"{name}: missing or invalid weight_scale")
            if scale.dtype != "U8":
                raise ValueError(f"{name}: invalid weight_scale dtype {scale.dtype}")
            if self.description.get(module + ".weight_scale") != scheme:
                raise ValueError(f"{name}: weight_scale scheme mismatch")
            dual_name = module + ".weight_dual_scale"
            if scheme == _MX4_DUAL and dual_name in self.tensors:
                dual = self.tensors[dual_name]
                if (
                    dual.dtype != "F32"
                    or len(dual.shape) != 3
                    or dual.shape[0] != n
                    or dual.shape[-1] != 1
                    or dual.shape[1] == 0
                    or groups % dual.shape[1]
                    or self.description.get(dual_name) != scheme
                ):
                    raise ValueError(f"{name}: invalid weight_dual_scale")
            elif scheme == _MX4_DUAL:
                # Some deliveries (e.g. the Ref2VA FP4 export) are converted without the
                # FP32 coarse dual scale and keep only the E8M0 group scale. Reconstruct
                # those as single-scale MXFP4 instead of rejecting the checkpoint; the
                # native MX runtime re-encodes to single-scale groups of 32 regardless.
                if dual_name in self.description:
                    raise ValueError(f"{name}: weight_dual_scale is declared but missing from the shards")
                self.single_scale_mx4 = True
            elif scheme == _MX4_SINGLE and dual_name in self.tensors:
                raise ValueError(f"{name}: single-scale MXFP4 must not provide weight_dual_scale")
        self.schemes = frozenset(
            self.description[name]
            for name in self.tensors
            if name.endswith(".weight") and self.description.get(name) != "FLOAT" and name in self.description
        )
        if not self.schemes:
            raise ValueError(f"offline checkpoint contains no quantized weights: {self.path}")
        if self.single_scale_mx4:
            logger.warning(
                "%s: FP4 weights carry only the E8M0 group scale (no weight_dual_scale); "
                "reconstructing single-scale MXFP4",
                self.path,
            )

    @property
    def diffusers_weights(self) -> bool:
        return any(name.startswith("transformer_blocks.") for name in self.tensors)

    def iter_weights(self, *, dequantize: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        with ExitStack() as stack:
            readers = {
                file: stack.enter_context(safe_open(str(file), framework="pt", device="cpu"))
                for file in {spec.file for spec in self.tensors.values()}
            }

            def read(name: str) -> torch.Tensor:
                return readers[self.tensors[name].file].get_tensor(name)

            for name in self.tensors:
                scheme = self.description.get(name, "FLOAT")
                kind = name.rpartition(".")[2]
                if kind == "bias":
                    scheme = "FLOAT"
                if scheme != "FLOAT" and kind != "weight":
                    continue
                weight = read(name)
                if dequantize and scheme != "FLOAT":
                    prefix = name.removesuffix("weight")
                    scale = read(prefix + "weight_scale")
                    dual = (
                        read(prefix + "weight_dual_scale")
                        if scheme == _MX4_DUAL and prefix + "weight_dual_scale" in self.tensors
                        else None
                    )
                    weight = reconstruct_mx_weight(weight, scale, dual)
                elif not dequantize and scheme != "FLOAT":
                    raise ValueError("MX checkpoint streaming requires load-time reconstruction (dequantize=True)")
                yield name, weight


def resolve_offline_components(
    config: Mapping[str, object] | None,
) -> tuple[dict[str, OfflineComponentCheckpoint], bool]:
    """Resolve the supplied delivery root or explicit component paths."""
    if config is None:
        return {}, False
    if not isinstance(config, Mapping):
        raise ValueError("offline_quantization must be a mapping")
    unknown = set(config) - {"path", "components", "execution_mode"}
    if unknown:
        raise ValueError(f"unknown offline_quantization options: {sorted(unknown)}")
    mode = config.get("execution_mode", "native_auto")
    if mode not in {"native", "native_auto", "dequantize", "native_mx"}:
        raise ValueError("offline execution_mode must be native_auto, native_mx, or dequantize")
    if mode != "dequantize":
        family = resolve_npu_quant_family()
        if mode in {"native", "native_auto"}:
            mode = "native_" + family
        elif mode != "native_" + family:
            raise ValueError(f"{mode} is incompatible with this NPU (requires {family} weights)")
    if "path" in config and "components" in config:
        raise ValueError("specify offline path or components, not both")
    paths: dict[str, str | Path]
    if "path" in config:
        paths = {"transformer": Path(str(config["path"]))}
    else:
        supplied_paths = config.get("components")
        if not isinstance(supplied_paths, dict) or not supplied_paths:
            raise ValueError("offline_quantization requires path or nonempty components")
        paths = supplied_paths
    if unknown_components := set(paths) - {"transformer", "video_vae", "audio_vae"}:
        raise ValueError(f"unsupported offline components: {sorted(unknown_components)}")
    checkpoints = {name: OfflineComponentCheckpoint(path) for name, path in paths.items()}
    for checkpoint in checkpoints.values():
        checkpoint.execution_mode = str(mode)
    if mode == "native_mx" and any(not checkpoint.schemes <= _MX_SCHEMES for checkpoint in checkpoints.values()):
        raise ValueError("native_mx requires MXFP8/MXFP4 component checkpoints; format conversion is never implicit")
    return checkpoints, True


def load_offline_vae_weights(model: torch.nn.Module, checkpoint: OfflineComponentCheckpoint) -> None:
    """Replace the original VAE state before installing its inference patches."""
    state = model.state_dict(keep_vars=True)
    loaded: set[str] = set()
    with torch.no_grad():
        for name, weight in checkpoint.iter_weights(dequantize=True):
            if name.endswith(".weight_quantizer.bias"):
                # ModelSlim can serialize the same floating bias twice. Only
                # discard this known alias when its canonical bias is present
                # and identical; never mask an unknown or conflicting tensor.
                canonical = name.removesuffix("weight_quantizer.bias") + "bias"
                spec = checkpoint.tensors.get(canonical)
                if spec is None or canonical not in state:
                    raise ValueError(f"offline VAE bias alias lacks canonical target: {name}")
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    bias = reader.get_tensor(canonical)
                if weight.dtype != bias.dtype or weight.shape != bias.shape or not torch.equal(weight, bias):
                    raise ValueError(f"offline VAE bias alias conflicts with canonical tensor: {name}")
                continue
            if name not in state and name.endswith((".weight_g", ".weight_v")):
                # PyTorch's parametrized weight_norm accepts the legacy g/v
                # names through load_state_dict hooks. Mirror that naming
                # conversion for this streaming copy without buffering a VAE.
                module, _, kind = name.rpartition(".")
                original = 0 if kind == "weight_g" else 1
                name = f"{module}.parametrizations.weight.original{original}"
            target = state.get(name)
            if target is None or target.shape != weight.shape:
                raise ValueError(f"offline VAE tensor does not match original architecture: {name}")
            if name in loaded:
                raise ValueError(f"duplicate offline VAE target: {name}")
            target.copy_(weight)
            loaded.add(name)
    if missing := state.keys() - loaded:
        raise ValueError(f"offline VAE checkpoint is incomplete: {sorted(missing)[:8]}")
