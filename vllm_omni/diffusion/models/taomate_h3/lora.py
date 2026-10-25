# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native TaoMate LoRA loading and optional retained student weights.

The artifact stores fused ``<target>.lora_a/b`` matrices over four projections
in each DiT/refiner block. Its QKV B rows already use the loaded [Q; K; V]
order. The shared diffusion LoRA manager applies these pairs and owns mode
switching; the optional bank preserves separate base/student storage for the
clean audio teacher, including graphs captured against the base parameters.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import SafetensorError, safe_open
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

TAOMATE_LORA_TARGET_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
TAOMATE_LORA_WEIGHTS_FILE = "adapter_model.safetensors"


def taomate_lora_targets(*, num_blocks: int = 50, num_refiner_blocks: int = 2) -> tuple[str, ...]:
    """The adapter's exact block x projection inventory, in checkpoint order."""
    blocks = [f"token_refiner.blocks.{index}" for index in range(num_refiner_blocks)]
    blocks.extend(f"blocks.{index}" for index in range(num_blocks))
    return tuple(f"{block}.{suffix}" for block in blocks for suffix in TAOMATE_LORA_TARGET_SUFFIXES)


class TaoMateLoRAError(RuntimeError):
    pass


def _read_adapter_config(adapter_dir: Path) -> tuple[int, float]:
    config_path = adapter_dir / "config.json"
    if not config_path.is_file():
        config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise TaoMateLoRAError(f"TaoMate-H3 adapter config not found under {adapter_dir}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaoMateLoRAError(f"cannot read TaoMate-H3 adapter config: {exc}") from exc
    rank = config.get("rank", config.get("r"))
    alpha = config.get("alpha", config.get("lora_alpha"))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise TaoMateLoRAError("TaoMate-H3 adapter rank is invalid")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or alpha <= 0:
        raise TaoMateLoRAError("TaoMate-H3 adapter alpha is invalid")
    return int(rank), float(alpha)


def prepare_taomate_lora(
    adapter_dir: str | Path,
    *,
    transformer: MiniMaxH3DiTModel,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    merge: bool = False,
) -> tuple[LoRAModel, PEFTHelper, MergedStudentWeights | None]:
    """Normalize native factors before final quantization, without installing LoRA.

    The model hook clones the returned model with the manager's request ID.
    A/B stay on CPU for the manager cache, rounded before either use; alpha/r stays
    separate for the fused addmm epilogue and FP32 merged-weight calculation.
    """
    root = Path(adapter_dir).expanduser().resolve()
    rank, alpha = _read_adapter_config(root)
    targets = taomate_lora_targets(
        num_blocks=transformer.arch.num_layers,
        num_refiner_blocks=transformer.arch.token_refiner_num_layers,
    )
    expected = {f"{target}.{leaf}" for target in targets for leaf in ("lora_a", "lora_b")}
    modules = dict(transformer.named_modules())
    loras: dict[str, LoRALayerWeights] = {}
    try:
        checkpoint = safe_open(str(root / TAOMATE_LORA_WEIGHTS_FILE), framework="pt", device="cpu")
    except (OSError, SafetensorError) as exc:
        raise TaoMateLoRAError(f"cannot read TaoMate-H3 adapter weights under {root}: {exc}") from exc
    with checkpoint as handle:
        present = set(handle.keys())
        if present != expected:
            raise TaoMateLoRAError(
                "TaoMate-H3 LoRA tensor inventory differs from the block x projection contract: "
                f"missing={len(expected - present)}, unexpected={len(present - expected)}"
            )
        for target in targets:
            module = modules.get(target)
            if module is None or not hasattr(module, "weight"):
                raise TaoMateLoRAError(f"TaoMate-H3 LoRA target {target!r} has no linear module in the DiT")
            a, b = (handle.get_tensor(f"{target}.{leaf}") for leaf in ("lora_a", "lora_b"))
            out_features, in_features = _linear_shape(module)
            if tuple(a.shape) != (rank, in_features) or tuple(b.shape) != (out_features, rank):
                raise TaoMateLoRAError(
                    f"TaoMate-H3 LoRA {target}: A {tuple(a.shape)} / B {tuple(b.shape)} do not match "
                    f"rank {rank} and linear ({out_features}, {in_features}); tensor parallel is unsupported"
                )
            name = f"transformer.{target}"
            loras[name] = LoRALayerWeights(
                module_name=name,
                rank=rank,
                lora_alpha=alpha,  # type: ignore[arg-type] - vLLM also accepts fractional alpha at runtime.
                lora_a=a.to(dtype=dtype).contiguous(),
                lora_b=b.to(dtype=dtype).contiguous(),
                scaling=alpha / rank,
            )
    model = LoRAModel(lora_model_id=1, rank=rank, loras=loras)
    peft = PEFTHelper.from_dict({"r": rank, "lora_alpha": alpha, "target_modules": list(loras)})
    bank = MergedStudentWeights(transformer, model) if merge else None
    return model, peft, bank


class MergedStudentWeights:
    """A parameter bank controlled by DiffusionLoRAManager, not a second adapter.

    Preparation registers shadows while the ordinary loader can still quantize
    them. ``bind`` runs only after final quantization and wrapper installation;
    it records the resulting parameters and buffers without copying their data.
    """

    def __init__(self, transformer: nn.Module, model: LoRAModel) -> None:
        self._shadows: dict[str, tuple[nn.Module, nn.Module]] = {}
        self._base_params: dict[str, dict[str, torch.Tensor]] = {}
        self._student_params: dict[str, dict[str, torch.Tensor]] = {}
        try:
            for name, weights in model.loras.items():
                module = transformer.get_submodule(name.removeprefix("transformer."))
                out_features, in_features = _linear_shape(module)
                # Upload the rounded factors before widening, one target at a
                # time. The manager keeps only their CPU copies in its cache.
                device = module.weight.device
                a = weights.lora_a.to(device=device, non_blocking=True).float()
                b = weights.lora_b.to(device=device, non_blocking=True).float()
                delta = (b @ a) * float(weights.scaling)
                if tuple(delta.shape) != (out_features, in_features):
                    raise TaoMateLoRAError(f"{name}: LoRA delta shape differs from the linear")
                weight = module.weight.detach()
                quantized = weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                if quantized:
                    base = _dequantize_linear_weight(module, out_features, in_features)
                    merged = (base + delta.to(base.device)).to(torch.bfloat16)
                else:
                    merged = (weight.to(delta.device, torch.float32) + delta).to(weight.dtype)
                shadow = _shadow_linear(module, merged)
                if hasattr(module, "taomate_student"):
                    raise TaoMateLoRAError(f"{name}: student weights have already been prepared")
                module.add_module("taomate_student", shadow)
                self._shadows[name] = (module, shadow)
                if quantized:
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is None or not hasattr(quant_method, "process_weights_after_loading"):
                        raise TaoMateLoRAError(f"{name}: quantized base weight without a quantization method")
                    quant_method.process_weights_after_loading(shadow)
                    if shadow.weight.dtype != weight.dtype or tuple(shadow.weight.shape) != tuple(weight.shape):
                        raise TaoMateLoRAError(f"{name}: requantized student weight differs from the base layout")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _param_set(module: nn.Module) -> dict[str, torch.Tensor]:
        return dict((*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)))

    def bind(self, modules: Mapping[str, nn.Module]) -> MergedStudentWeights:
        """Validate complete bindings and snapshot finalized base/student storage."""
        if not self._shadows:
            raise TaoMateLoRAError("cannot bind a closed student weight bank")
        if self._base_params:
            return self
        base_params, student_params = {}, {}
        for name, (module, shadow) in self._shadows.items():
            wrapper = modules.get(name)
            if wrapper is None or getattr(wrapper, "base_layer", wrapper) is not module:
                raise TaoMateLoRAError(f"{name}: merged student did not bind to its original linear")
            student = self._param_set(shadow)
            base = {key: getattr(module, key) for key in student if hasattr(module, key)}
            if set(student) != set(base):
                raise TaoMateLoRAError(f"{name}: student parameters have no base counterpart")
            if any(student[key].dtype != base[key].dtype or student[key].shape != base[key].shape for key in student):
                raise TaoMateLoRAError(f"{name}: student and base parameters differ after quantization")
            base_params[name], student_params[name] = base, student
        self._base_params, self._student_params = base_params, student_params
        return self

    def _install(self, params: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, values in params.items():
            module, _ = self._shadows[name]
            for key, value in values.items():
                setattr(module, key, value)

    def activate(self) -> None:
        if not self._student_params:
            raise TaoMateLoRAError("student weight bank must be bound after quantization before activation")
        self._install(self._student_params)

    def suspend(self) -> None:
        self._install(self._base_params)

    def close(self) -> None:
        """Restore the base and discard owned shadows; safe after failed setup."""
        self.suspend()
        for module, shadow in self._shadows.values():
            if getattr(module, "taomate_student", None) is shadow:
                delattr(module, "taomate_student")
        self._shadows.clear()
        self._base_params.clear()
        self._student_params.clear()


def _dequantize_linear_weight(module: nn.Module, out_features: int, in_features: int) -> torch.Tensor:
    """``[out, in]`` float32 base weight of an online-quantized vLLM linear.

    Online FP8 methods store the quantized weight either as ``[out, in]`` or
    transposed as ``[in, out]`` (the Cutlass kernels want the latter) with a
    per-tensor scale (one element) or a per-channel scale (``[out, 1]``).
    """
    weight = module.weight.detach()
    if tuple(weight.shape) == (in_features, out_features) and in_features != out_features:
        weight = weight.t()
    elif tuple(weight.shape) != (out_features, in_features):
        raise TaoMateLoRAError(
            f"unexpected quantized weight shape {tuple(weight.shape)} for ({out_features}, {in_features})"
        )
    scale = getattr(module, "weight_scale", None)
    if scale is None:
        raise TaoMateLoRAError("quantized linear without weight_scale")
    scale = scale.detach().to(torch.float32)
    base = weight.to(torch.float32)
    if scale.numel() == 1:
        return base * scale.reshape(())
    if scale.numel() == out_features:
        return base * scale.reshape(out_features, 1)
    raise TaoMateLoRAError(f"unsupported weight_scale shape {tuple(scale.shape)} for ({out_features}, {in_features})")


def _shadow_linear(module: nn.Module, merged_weight: torch.Tensor) -> nn.Module:
    """A shallow twin of a vLLM linear that owns ``merged_weight`` and shares everything else.

    vLLM's parameter classes cannot be deep-copied, so the twin copies the
    module's attribute dictionary, gets fresh parameter/buffer/hook registries,
    a plain Parameter for the weight (the bias, if any, is shared) and the
    module's quantization method, and is still unprocessed so the loader's
    online quantization treats it like the base linear.
    """
    shadow = object.__new__(type(module))
    shadow.__dict__.update(module.__dict__)
    shadow._parameters = {}
    shadow._buffers = {}
    shadow._non_persistent_buffers_set = set()
    shadow._modules = {}
    for name, value in module.__dict__.items():
        if name.endswith("hooks") and isinstance(value, dict):
            shadow.__dict__[name] = type(value)()
    shadow._parameters["weight"] = nn.Parameter(merged_weight.contiguous(), requires_grad=False)
    bias = module._parameters.get("bias")
    if bias is not None:
        shadow._parameters["bias"] = bias
    for name, buffer in module._buffers.items():
        shadow._buffers[name] = buffer
    shadow._already_called_process_weights_after_loading = False
    return shadow


def _linear_shape(module: nn.Module) -> tuple[int, int]:
    """(out_features, in_features) of a vLLM linear, robust to quantized layouts."""
    out_features = getattr(module, "output_size", None)
    in_features = getattr(module, "input_size", None)
    if out_features is None or in_features is None:
        weight = module.weight
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    return int(out_features), int(in_features)


__all__ = [
    "MergedStudentWeights",
    "TAOMATE_LORA_TARGET_SUFFIXES",
    "TaoMateLoRAError",
    "prepare_taomate_lora",
    "taomate_lora_targets",
]
