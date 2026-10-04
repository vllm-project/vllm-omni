# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-H3 native LoRA adapter applied as switchable forward hooks.

The TaoMate-H3 release (``TaoLiveAIGC/TaoMate-H3``) is a rank-128 LoRA over
the fused MiniMax-H3 checkpoint layout: 52 blocks (2 token-refiner blocks and
50 DiT blocks) x 4 projections (``attn.qkv_proj``, ``attn.out_proj``,
``mlp.fc1``, ``mlp.fc2``), stored as ``<target>.lora_a`` [r, in] and
``<target>.lora_b`` [out, r] in ``adapter_model.safetensors`` with
``{"rank": 128, "alpha": 128.0}`` in ``config.json`` / ``adapter_config.json``.

The adapter is deliberately *not* merged into the weights: TaoMate's audio
teacher is the base H3 model without the LoRA, and it must share the resident
BF16 weights with the LoRA student (two copies of the 62 GB DiT do not fit one
device). Instead every target linear gets a forward hook that adds
``(x @ A^T) @ B^T * alpha / rank`` to its output while the adapter is enabled;
``disabled()`` switches it off for the teacher forwards.

Only ``tensor_parallel_size == 1`` is supported (the realtime topology is pure
Ulysses sequence parallelism): the adapter is then replicated on every rank
and no A/B sharding is required. The fused QKV base weight is re-ordered from
the checkpoint's per-head grouped layout to ``[Q; K; V]`` at load time; the
adapter's fused-QKV ``lora_b`` already uses that merged row order (the release
runtime applies it unpermuted), so it is consumed as stored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from vllm.logger import init_logger

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

logger = init_logger(__name__)

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


def is_taomate_lora_dir(path: str | Path | None) -> bool:
    """Whether ``path`` looks like a TaoMate-H3 native adapter directory."""
    if not path:
        return False
    root = Path(path)
    if not root.is_dir() or not (root / TAOMATE_LORA_WEIGHTS_FILE).is_file():
        return False
    if not ((root / "config.json").is_file() or (root / "adapter_config.json").is_file()):
        return False
    try:
        with safe_open(str(root / TAOMATE_LORA_WEIGHTS_FILE), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception:  # noqa: BLE001 - not ours if it does not open
        return False
    return any(key.endswith(".lora_a") for key in keys) and any(key.endswith(".lora_b") for key in keys)


class TaoMateLoRAAdapter:
    """Hook-based, switchable application of the TaoMate-H3 LoRA."""

    def __init__(
        self,
        *,
        rank: int,
        alpha: float,
        lora_a: dict[str, torch.Tensor],
        lora_b: dict[str, torch.Tensor],
        source: str,
    ) -> None:
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.source = source
        self._lora_a = lora_a
        self._lora_b = lora_b
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._bound_targets: tuple[str, ...] = ()
        self.enabled = True
        # Merged student weights (see build_merged_student): per target the
        # module and its shadow linear holding W + scale * B @ A, plus the
        # parameter sets swapped in for the student and for the base model.
        self._shadows: dict[str, tuple[nn.Module, nn.Module]] = {}
        self._base_params: dict[str, dict[str, torch.Tensor]] = {}
        self._student_params: dict[str, dict[str, torch.Tensor]] = {}
        self._student_installed = False

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self._lora_a)

    @property
    def bound_targets(self) -> tuple[str, ...]:
        return self._bound_targets

    @classmethod
    def load(
        cls,
        adapter_dir: str | Path,
        *,
        transformer: MiniMaxH3DiTModel,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> TaoMateLoRAAdapter:
        """Read the adapter and prepare rank-local A/B buffers in the model's fused layout."""
        root = Path(adapter_dir).expanduser().resolve()
        rank, alpha = _read_adapter_config(root)
        arch = transformer.arch
        targets = taomate_lora_targets(
            num_blocks=arch.num_layers,
            num_refiner_blocks=arch.token_refiner_num_layers,
        )
        expected = {f"{target}.{leaf}" for target in targets for leaf in ("lora_a", "lora_b")}
        weights_path = root / TAOMATE_LORA_WEIGHTS_FILE
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}
        modules = dict(transformer.named_modules())
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            present = set(handle.keys())
            if present != expected:
                raise TaoMateLoRAError(
                    "TaoMate-H3 LoRA tensor inventory differs from the 52-block x 4-projection contract: "
                    f"missing={len(expected - present)}, unexpected={len(present - expected)}"
                )
            for target in targets:
                module = modules.get(target)
                if module is None or not hasattr(module, "weight"):
                    raise TaoMateLoRAError(f"TaoMate-H3 LoRA target {target!r} has no linear module in the DiT")
                a = handle.get_tensor(f"{target}.lora_a")
                b = handle.get_tensor(f"{target}.lora_b")
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
                    raise TaoMateLoRAError(f"TaoMate-H3 LoRA tensor shape differs for {target}")
                weight = module.weight
                out_features, in_features = _linear_shape(module)
                if a.shape[1] != in_features or b.shape[0] != out_features:
                    raise TaoMateLoRAError(
                        f"TaoMate-H3 LoRA {target}: A {tuple(a.shape)} / B {tuple(b.shape)} do not match the "
                        f"linear ({out_features}, {in_features}); tensor parallel is not supported by this adapter"
                    )
                # The adapter's fused-QKV ``lora_b`` is already in the merged
                # [Q; K; V] row order of the loaded base weight (the release
                # loader reorders only the grouped base checkpoint), so no
                # permutation is applied here.
                lora_a[target] = a.to(device=device, dtype=dtype).contiguous()
                lora_b[target] = b.to(device=device, dtype=dtype).contiguous()
                del weight
        adapter = cls(rank=rank, alpha=alpha, lora_a=lora_a, lora_b=lora_b, source=str(root))
        adapter.bind(transformer)
        logger.info(
            "TaoMate-H3 LoRA loaded from %s: rank=%d alpha=%g targets=%d dtype=%s",
            root,
            rank,
            alpha,
            len(targets),
            dtype,
        )
        return adapter

    def bind(self, transformer: nn.Module) -> None:
        """Install the forward hooks on every target linear."""
        self.unbind()
        modules = dict(transformer.named_modules())
        bound: list[str] = []
        for target in self.targets:
            module = modules[target]
            self._handles.append(module.register_forward_hook(self._make_hook(target)))
            bound.append(target)
        self._bound_targets = tuple(bound)

    def unbind(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bound_targets = ()

    def _make_hook(self, target: str):
        lora_a = self._lora_a[target]
        lora_b = self._lora_b[target]
        scale = float(self.scale)
        adapter = self

        lora_a_t = lora_a.t()
        lora_b_t = lora_b.t()

        def hook(module: nn.Module, args: tuple, output):
            if not adapter.enabled:
                return None
            x = args[0]
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            flat = x.reshape(-1, x.shape[-1]).to(lora_a.dtype)
            low_rank = torch.matmul(flat, lora_a_t)
            out_rows = out.reshape(-1, out.shape[-1])
            if out_rows.dtype == low_rank.dtype and out_rows.data_ptr() == out.data_ptr():
                # One GEMM with the scale and the residual add in its epilogue
                # (two launches per target instead of four; the sum rounds once).
                out_rows.addmm_(low_rank, lora_b_t, alpha=scale)
                return None
            delta = torch.matmul(low_rank, lora_b_t)
            if scale != 1.0:
                delta = delta * scale
            out.add_(delta.reshape(out.shape).to(out.dtype))
            return None

        return hook

    # -- merged student weights ------------------------------------------

    @property
    def merged(self) -> bool:
        return bool(self._shadows)

    def build_merged_student(self, transformer: nn.Module) -> int:
        """Give every target a shadow linear whose weight is ``W + scale * B @ A``.

        Must run while the base weights are still BF16, before the loader's
        online quantization: the shadow is registered as a submodule of its
        target (``<target>.taomate_student``) so the loader quantizes it with
        the target's own quantization method (per-channel FP8 keeps the delta
        at channel resolution). Afterwards the student forwards use the shadow's
        parameters and the hooks stay off; the base parameters are swapped back
        in for the teacher (``disabled()``). Returns the number of targets.
        """
        if self._shadows:
            return len(self._shadows)
        modules = dict(transformer.named_modules())
        for target in self.targets:
            module = modules[target]
            out_features, in_features = _linear_shape(module)
            delta = torch.matmul(self._lora_b[target].float(), self._lora_a[target].float()) * float(self.scale)
            if tuple(delta.shape) != (out_features, in_features):
                raise TaoMateLoRAError(
                    f"{target}: delta {tuple(delta.shape)} does not match the linear ({out_features}, {in_features})"
                )
            weight = module.weight.detach()
            quantized = weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            if quantized:
                # The loader quantized the base while loading it: dequantize
                # (per-tensor or per-channel scale, stored transposed or not),
                # merge in float32, and requantize the twin with the same method.
                base = _dequantize_linear_weight(module, out_features, in_features)
                merged = (base + delta.to(base.device)).to(torch.bfloat16)
            else:
                merged = (weight.to(delta.device, torch.float32) + delta).to(weight.dtype)
            shadow = _shadow_linear(module, merged)
            module.add_module("taomate_student", shadow)
            if quantized:
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None or not hasattr(quant_method, "process_weights_after_loading"):
                    raise TaoMateLoRAError(f"{target}: quantized base weight without a quantization method")
                quant_method.process_weights_after_loading(shadow)
                if shadow.weight.dtype != weight.dtype or tuple(shadow.weight.shape) != tuple(weight.shape):
                    raise TaoMateLoRAError(
                        f"{target}: requantized student weight {shadow.weight.dtype} {tuple(shadow.weight.shape)} "
                        f"differs from the base {weight.dtype} {tuple(weight.shape)}"
                    )
            self._shadows[target] = (module, shadow)
        # The delta now lives in the student weights; hooks must not add it again.
        self.enabled = False
        self._student_installed = False
        return len(self._shadows)

    @staticmethod
    def _param_set(module: nn.Module) -> dict[str, torch.Tensor]:
        names = [name for name, _ in module.named_parameters(recurse=False)]
        names += [name for name, _ in module.named_buffers(recurse=False)]
        return {name: getattr(module, name) for name in names}

    def _snapshot_base(self) -> None:
        if self._base_params:
            return
        for target, (module, shadow) in self._shadows.items():
            student = self._param_set(shadow)
            base = {name: getattr(module, name) for name in student if hasattr(module, name)}
            missing = set(student) - set(base)
            if missing:
                raise TaoMateLoRAError(f"{target}: student parameters {sorted(missing)} have no base counterpart")
            if any(getattr(shadow, name).dtype != base[name].dtype for name in student):
                raise TaoMateLoRAError(f"{target}: student and base parameters differ in dtype (quantized alike?)")
            self._base_params[target] = base
            self._student_params[target] = student

    def _install(self, params: dict[str, dict[str, torch.Tensor]]) -> None:
        for target, (module, _) in self._shadows.items():
            for name, value in params[target].items():
                # Assigning a registered Parameter or buffer swaps it in place.
                setattr(module, name, value)

    def ensure_student(self) -> None:
        """Install the merged student weights (no-op without a merge or when installed)."""
        if not self._shadows or self._student_installed:
            return
        self._snapshot_base()
        self._install(self._student_params)
        self._student_installed = True

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Run the base model without the adapter (the audio teacher path)."""
        if self._shadows:
            # Base weights for the teacher, student weights back afterwards.
            self._snapshot_base()
            self._install(self._base_params)
            self._student_installed = False
            try:
                yield
            finally:
                self._install(self._student_params)
                self._student_installed = True
            return
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._lora_a.values()) + sum(
            t.numel() * t.element_size() for t in self._lora_b.values()
        )


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
    "TAOMATE_LORA_TARGET_SUFFIXES",
    "TaoMateLoRAAdapter",
    "TaoMateLoRAError",
    "is_taomate_lora_dir",
    "taomate_lora_targets",
]
