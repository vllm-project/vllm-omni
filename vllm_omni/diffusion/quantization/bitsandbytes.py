# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextvars
import fnmatch
import os
import threading
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from vllm.logger import init_logger

from .base import DiffusionQuantizationConfig

BnbBackend = Literal["bitsandbytes_8bit", "bitsandbytes_4bit"]
DEFAULT_BNB_MODULES = ("transformer", "text_encoder*")

logger = init_logger(__name__)

_BNB_LOAD_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "diffusion_bnb_load_context",
    default=None,
)
_BNB_PATCH_LOCK = threading.Lock()
_BNB_PATCH_APPLIED = False
_BNB_PATCH_REFCOUNT = 0
_BNB_PATCH_ORIG = None
_BNB_PATCH_TARGET = None


@dataclass
class _BnbPipelineState:
    quantized_components: set[str] = field(default_factory=set)
    offload_skip_components: set[str] = field(default_factory=set)


_BNB_PIPELINE_STATE: "weakref.WeakKeyDictionary[Any, _BnbPipelineState]" = weakref.WeakKeyDictionary()


def _get_bnb_pipeline_state(pipeline: Any) -> _BnbPipelineState:
    try:
        state = _BNB_PIPELINE_STATE.get(pipeline)
    except TypeError:
        state = getattr(pipeline, "_bnb_pipeline_state", None)
        if state is None:
            state = _BnbPipelineState()
            try:
                setattr(pipeline, "_bnb_pipeline_state", state)
            except Exception:
                pass
        return state
    if state is None:
        state = _BnbPipelineState()
        _BNB_PIPELINE_STATE[pipeline] = state
    return state


def get_bnb_quantized_components(pipeline: Any) -> set[str]:
    return set(_get_bnb_pipeline_state(pipeline).quantized_components)


def set_bnb_quantized_components(pipeline: Any, components: set[str] | Iterable[str]) -> None:
    state = _get_bnb_pipeline_state(pipeline)
    state.quantized_components = set(components)


def update_bnb_quantized_components(pipeline: Any, components: Iterable[str]) -> None:
    state = _get_bnb_pipeline_state(pipeline)
    state.quantized_components.update(components)


def get_bnb_offload_skip_components(pipeline: Any) -> set[str]:
    return set(_get_bnb_pipeline_state(pipeline).offload_skip_components)


def set_bnb_offload_skip_components(pipeline: Any, components: Iterable[str]) -> None:
    state = _get_bnb_pipeline_state(pipeline)
    state.offload_skip_components = set(components)


def _normalize_modules(modules: Sequence[str] | str | None) -> list[str] | None:
    if modules is None:
        return None
    if isinstance(modules, str):
        items = [m.strip() for m in modules.split(",")]
    else:
        items = [str(m).strip() for m in modules]
    return [m for m in items if m]


def matches_bnb_module_name(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _resolve_module_patterns(available: Iterable[str], patterns: Sequence[str]) -> list[str]:
    ordered_available = list(dict.fromkeys(available))
    matched: list[str] = []
    for pattern in patterns:
        for name in ordered_available:
            if fnmatch.fnmatchcase(name, pattern) and name not in matched:
                matched.append(name)
    return matched


def _get_pipeline_component_names(pipeline: Any) -> list[str]:
    names: list[str] = []
    if isinstance(pipeline, nn.Module):
        names.extend(pipeline._modules.keys())
    components = getattr(pipeline, "components", None)
    if isinstance(components, Mapping):
        for name in components.keys():
            if name not in names:
                names.append(name)
    return names


def _normalize_bnb_compute_dtype(value: torch.dtype | str | None) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        dtype_str = value.strip().lower()
        if dtype_str in ("", "auto"):
            return None
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        if dtype_str not in dtype_map:
            raise ValueError(
                f"Unknown bnb_4bit_compute_dtype {value!r}. Supported: {sorted(dtype_map.keys()) + ['auto']}"
            )
        return dtype_map[dtype_str]
    raise TypeError(f"bnb_4bit_compute_dtype must be a torch.dtype, str, or None (got {type(value)!r})")


@dataclass
class DiffusionBitsAndBytesConfig(DiffusionQuantizationConfig):
    """Diffusion bitsandbytes config aligned with vLLM bitsandbytes fields."""

    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype | str | None = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    modules: Sequence[str] | str | None = None

    def __post_init__(self) -> None:
        if self.load_in_8bit and self.load_in_4bit:
            # Prefer 8bit if both are set (avoid ambiguous defaults).
            self.load_in_4bit = False
        if not self.load_in_8bit and not self.load_in_4bit:
            raise ValueError("bitsandbytes config requires load_in_8bit or load_in_4bit to be True")
        self.bnb_4bit_compute_dtype = _normalize_bnb_compute_dtype(self.bnb_4bit_compute_dtype)
        self.modules = _normalize_modules(self.modules)
        self._vllm_config = None

    @classmethod
    def get_name(cls) -> str:
        return "bitsandbytes"

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_backend(self) -> BnbBackend:
        return "bitsandbytes_8bit" if self.load_in_8bit else "bitsandbytes_4bit"

    def get_modules(self) -> list[str]:
        if self.modules:
            return list(self.modules)
        return list(DEFAULT_BNB_MODULES)


def get_bnb_module_kwargs(
    quant_config: DiffusionBitsAndBytesConfig | None,
    module_name: str | None,
    device: torch.device,
    *,
    enable_cpu_offload: bool | None = None,
) -> dict[str, Any]:
    """Build kwargs for transformers.from_pretrained to enable bnb quant at load time."""
    if quant_config is None:
        return {}

    if not module_name:
        return {}

    if not matches_bnb_module_name(module_name, quant_config.get_modules()):
        return {}

    if device.type != "cuda":
        return {}

    if not torch.cuda.is_available():
        return {}

    try:
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
    except Exception:
        return {}

    backend = quant_config.get_backend()
    if backend == "bitsandbytes_8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(enable_cpu_offload),
            llm_int8_has_fp16_weight=bool(quant_config.llm_int8_has_fp16_weight),
        )
    else:
        compute_dtype = quant_config.bnb_4bit_compute_dtype or torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
            llm_int8_enable_fp32_cpu_offload=bool(enable_cpu_offload),
            llm_int8_has_fp16_weight=bool(quant_config.llm_int8_has_fp16_weight),
        )

    return {
        "device_map": {"": str(device)},
        "low_cpu_mem_usage": True,
        "quantization_config": bnb_config,
    }


def _infer_bnb_module_name(
    model_name_or_path: Any,
    subfolder: str | None,
) -> str | None:
    if subfolder:
        return str(subfolder)
    if isinstance(model_name_or_path, (str, os.PathLike)):
        base = os.path.basename(str(model_name_or_path).rstrip("/"))
        if matches_bnb_module_name(base, DEFAULT_BNB_MODULES):
            return base
    return None


def _ensure_transformers_bnb_patch() -> bool:
    """Ensure a thread-safe from_pretrained hook is installed.

    The hook is a no-op unless a context var is set by
    patch_transformers_for_bnb_load.
    """
    global _BNB_PATCH_APPLIED, _BNB_PATCH_REFCOUNT, _BNB_PATCH_ORIG, _BNB_PATCH_TARGET
    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore[import-not-found]
    except Exception:
        return False

    with _BNB_PATCH_LOCK:
        if _BNB_PATCH_APPLIED:
            _BNB_PATCH_REFCOUNT += 1
            return True

        orig_attr = PreTrainedModel.__dict__.get("from_pretrained")
        if orig_attr is None:
            return False

        orig_func = orig_attr.__func__

        def _wrapped_from_pretrained(cls, model_name_or_path, *args, **kwargs):  # type: ignore[no-untyped-def]
            ctx = _BNB_LOAD_CONTEXT.get()
            if ctx is None:
                return orig_func(cls, model_name_or_path, *args, **kwargs)

            module_name = _infer_bnb_module_name(model_name_or_path, kwargs.get("subfolder"))
            bnb_kwargs = get_bnb_module_kwargs(
                ctx["quant_config"],
                module_name,
                ctx["device"],
                enable_cpu_offload=ctx.get("enable_cpu_offload"),
            )
            if bnb_kwargs:
                merged_kwargs = dict(kwargs)
                for key, value in bnb_kwargs.items():
                    merged_kwargs.setdefault(key, value)
                kwargs = merged_kwargs
                if module_name is not None:
                    ctx["quantized_components"].add(module_name)
            return orig_func(cls, model_name_or_path, *args, **kwargs)

        PreTrainedModel.from_pretrained = classmethod(_wrapped_from_pretrained)
        _BNB_PATCH_APPLIED = True
        _BNB_PATCH_REFCOUNT = 1
        _BNB_PATCH_ORIG = orig_attr
        _BNB_PATCH_TARGET = PreTrainedModel
        return True


def _release_transformers_bnb_patch() -> None:
    global _BNB_PATCH_APPLIED, _BNB_PATCH_REFCOUNT, _BNB_PATCH_ORIG, _BNB_PATCH_TARGET
    with _BNB_PATCH_LOCK:
        if not _BNB_PATCH_APPLIED:
            return
        _BNB_PATCH_REFCOUNT = max(0, _BNB_PATCH_REFCOUNT - 1)
        if _BNB_PATCH_REFCOUNT > 0:
            return
        if _BNB_PATCH_TARGET is not None and _BNB_PATCH_ORIG is not None:
            _BNB_PATCH_TARGET.from_pretrained = _BNB_PATCH_ORIG
        _BNB_PATCH_APPLIED = False
        _BNB_PATCH_ORIG = None
        _BNB_PATCH_TARGET = None


@contextmanager
def patch_transformers_for_bnb_load(
    quant_config: DiffusionBitsAndBytesConfig | None,
    *,
    device: torch.device,
    enable_cpu_offload: bool | None = None,
    enable_hf_bnb_load: bool = True,
) -> Iterator[set[str]]:
    """Temporarily inject bitsandbytes kwargs into transformers.from_pretrained.

    Returns a set of component names that were loaded with bitsandbytes config.
    """
    if quant_config is None or not enable_hf_bnb_load:
        yield set()
        return

    if device.type != "cuda":
        yield set()
        return

    if not _ensure_transformers_bnb_patch():
        yield set()
        return

    quantized_components: set[str] = set()
    ctx = {
        "quant_config": quant_config,
        "device": device,
        "enable_cpu_offload": enable_cpu_offload,
        "quantized_components": quantized_components,
    }
    token = _BNB_LOAD_CONTEXT.set(ctx)
    try:
        yield quantized_components
    finally:
        _BNB_LOAD_CONTEXT.reset(token)
        _release_transformers_bnb_patch()


def apply_bnb_quantization(
    pipeline: nn.Module,
    quant_config: DiffusionBitsAndBytesConfig | None,
    *,
    copy_weights: bool = True,
    only_modules: Iterable[str] | None = None,
    skip_modules: Iterable[str] | None = None,
) -> set[str]:
    """Apply bitsandbytes weight-only quantization to selected pipeline components.

    This function is best-effort:
    - Only replaces `torch.nn.Linear` modules (MVP scope).
    - Skips missing component names in the configured module list.

    Returns:
        Set of component names that were quantized.
    """

    if quant_config is None:
        return set()
    quant_backend = quant_config.get_backend()

    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for diffusion quantization='bitsandbytes'. "
            "Install with: `pip install bitsandbytes`."
        ) from exc

    if quant_config.llm_int8_enable_fp32_cpu_offload:
        logger.warning_once(
            "llm_int8_enable_fp32_cpu_offload only applies to HF load-time quantization; "
            "it is ignored for post-hoc quantization."
        )

    requested_modules = quant_config.get_modules()
    available_modules = list(only_modules) if only_modules is not None else _get_pipeline_component_names(pipeline)
    quant_modules = _resolve_module_patterns(available_modules, requested_modules)
    if skip_modules is not None:
        skip = set(skip_modules)
        quant_modules = [m for m in quant_modules if m not in skip]
    if not quant_modules:
        if only_modules is None:
            logger.warning_once(
                "bitsandbytes: none of the configured modules were found on the pipeline (%s).",
                tuple(requested_modules),
            )
        return set()

    logger.info_once("Applying bitsandbytes quantization to modules=%s", tuple(quant_modules))

    bnb_compute_dtype = quant_config.bnb_4bit_compute_dtype or torch.float32
    quantized_components: set[str] = set()
    replaced_any = False

    for module_name in quant_modules:
        component = getattr(pipeline, module_name, None)
        if component is None:
            continue
        if not isinstance(component, nn.Module):
            continue
        already_quantized = _contains_bnb_linear(component, bnb)
        num_replaced = _apply_to_component(
            component,
            bnb=bnb,
            backend=quant_backend,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
            llm_int8_has_fp16_weight=quant_config.llm_int8_has_fp16_weight,
            copy_weights=copy_weights,
        )
        if num_replaced > 0:
            quantized_components.add(module_name)
            replaced_any = True
        elif already_quantized:
            quantized_components.add(module_name)
            replaced_any = True
        else:
            logger.warning_once(
                "bitsandbytes: no Linear layers replaced in module '%s' (%s).",
                module_name,
                component.__class__.__name__,
            )

    if not replaced_any:
        logger.warning_once("bitsandbytes: no Linear layers replaced; quantization may be ineffective.")

    return quantized_components


def _apply_to_component(
    component: nn.Module,
    *,
    bnb: Any,
    backend: BnbBackend,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: torch.dtype | str | None,
    bnb_4bit_use_double_quant: bool,
    llm_int8_has_fp16_weight: bool,
    copy_weights: bool,
) -> int:
    original_device = _get_module_device(component)

    # Linear4bit requires fp weights before .to("cuda") triggers internal packing.
    # We avoid migrating the whole component; per-layer CPU copies are handled in _load_linear_weights.

    num_replaced = _replace_linear_modules_inplace(
        component,
        bnb=bnb,
        backend=backend,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
        copy_weights=copy_weights,
    )
    logger.info("Quantized %d Linear layers with %s in %s", num_replaced, backend, component.__class__.__name__)
    if original_device is not None and original_device.type != "cpu":
        component.to(original_device)
    return num_replaced


def _replace_linear_modules_inplace(
    root: nn.Module,
    *,
    bnb: Any,
    backend: BnbBackend,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: torch.dtype | str | None,
    bnb_4bit_use_double_quant: bool,
    llm_int8_has_fp16_weight: bool,
    copy_weights: bool,
) -> int:
    replaced = 0
    for child_name, child in list(root.named_children()):
        if _is_bnb_linear(child, bnb):
            continue
        if _is_vllm_linear(child) and backend == "bitsandbytes_4bit" and copy_weights:
            if _quantize_vllm_linear_inplace(
                child,
                bnb=bnb,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            ):
                replaced += 1
            continue
        if _is_supported_linear(child, copy_weights=copy_weights):
            new_child = _convert_linear_module(
                child,
                bnb=bnb,
                backend=backend,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
                copy_weights=copy_weights,
            )
            _set_child_module(root, child_name, new_child)
            replaced += 1
            continue

        replaced += _replace_linear_modules_inplace(
            child,
            bnb=bnb,
            backend=backend,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            copy_weights=copy_weights,
        )

    return replaced


class _DiffusionBnbLinearMethod:
    """Minimal bnb 4bit method for vLLM Linear modules."""

    def __init__(self, compute_dtype: torch.dtype | None) -> None:
        self.compute_dtype = compute_dtype

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from bitsandbytes import matmul_4bit

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True

        compute_dtype = self.compute_dtype or x.dtype
        if compute_dtype != x.dtype:
            x = x.to(compute_dtype)

        weight = layer.weight
        out = matmul_4bit(x, weight.t(), weight.quant_state)
        if out.dtype != original_type:
            out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias
        return out


def _quantize_vllm_linear_inplace(
    linear: nn.Module,
    *,
    bnb: Any,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: torch.dtype | str | None,
    bnb_4bit_use_double_quant: bool,
) -> bool:
    if getattr(linear, "tp_size", 1) != 1:
        return False
    weight = getattr(linear, "weight", None)
    if weight is None:
        return False
    if getattr(weight, "quant_state", None) is not None:
        return False
    original_device = weight.device
    if original_device.type != "cuda":
        return False

    in_features, out_features = _get_linear_io_features(linear)
    compute_dtype = bnb_4bit_compute_dtype or torch.float32

    temp = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=compute_dtype,
        compress_statistics=bnb_4bit_use_double_quant,
        quant_type=bnb_4bit_quant_type,
        device=torch.device("cpu"),
    )
    _load_linear_weights(linear, temp, include_bias=False)
    temp = temp.to(original_device)

    linear._parameters["weight"] = temp.weight
    linear.quant_method = _DiffusionBnbLinearMethod(
        compute_dtype=getattr(temp, "compute_dtype", compute_dtype),
    )
    return True


def _convert_linear_module(
    linear: nn.Module,
    *,
    bnb: Any,
    backend: BnbBackend,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: torch.dtype | str | None,
    bnb_4bit_use_double_quant: bool,
    llm_int8_has_fp16_weight: bool,
    copy_weights: bool,
) -> nn.Module:
    original_device = linear.weight.device
    in_features, out_features = _get_linear_io_features(linear)
    bias = getattr(linear, "bias", None)
    has_bias = bias is not None
    is_vllm_linear = _is_vllm_linear(linear)
    return_bias = bool(getattr(linear, "return_bias", False))
    skip_bias_add = bool(getattr(linear, "skip_bias_add", False))

    if copy_weights:
        target_device = torch.device("cpu")
    else:
        target_device = original_device

    # Bias handling truth table (only affects inner module bias allocation):
    # - return_bias=False, skip_bias_add=False -> keep bias
    # - return_bias=False, skip_bias_add=True  -> keep bias (caller ignores)
    # - return_bias=True,  skip_bias_add=False -> keep bias (bias added inside)
    # - return_bias=True,  skip_bias_add=True  -> drop bias (bias returned separately)
    inner_has_bias = has_bias if not (is_vllm_linear and return_bias and skip_bias_add) else False
    if backend == "bitsandbytes_8bit":
        new_linear = bnb.nn.Linear8bitLt(
            in_features,
            out_features,
            bias=inner_has_bias,
            has_fp16_weights=llm_int8_has_fp16_weight,
            device=target_device,
        )
    elif backend == "bitsandbytes_4bit":
        new_linear = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=inner_has_bias,
            compute_dtype=bnb_4bit_compute_dtype,
            compress_statistics=bnb_4bit_use_double_quant,
            quant_type=bnb_4bit_quant_type,
            device=target_device,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if copy_weights:
        _load_linear_weights(linear, new_linear, include_bias=inner_has_bias)
        if original_device.type != "cpu":
            new_linear = new_linear.to(original_device)

    if is_vllm_linear and return_bias:
        bias_param = None
        if has_bias and skip_bias_add:
            bias_param = bias.detach().clone()
        return _BnbLinearReturnBiasWrapper(
            linear=new_linear,
            bias=bias_param,
            return_bias=return_bias,
            skip_bias_add=skip_bias_add,
            meta=_collect_vllm_linear_meta(linear),
        )

    return new_linear


def _is_bnb_linear(module: nn.Module, bnb: Any) -> bool:
    return isinstance(module, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit))


def _contains_bnb_linear(module: nn.Module, bnb: Any) -> bool:
    return any(_is_bnb_linear(child, bnb) for child in module.modules())


def _is_vllm_linear(module: nn.Module) -> bool:
    module_path = getattr(module.__class__, "__module__", "")
    if module_path.startswith("vllm.model_executor.layers.linear"):
        return True
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except Exception:
        return False
    return isinstance(module, LinearBase)


def _is_supported_linear(module: nn.Module, *, copy_weights: bool) -> bool:
    if isinstance(module, nn.Linear):
        return True
    if not _is_vllm_linear(module):
        return False
    # Avoid pre-replace for vLLM linear modules to keep weight loading safe.
    if not copy_weights:
        return False
    tp_size = getattr(module, "tp_size", 1)
    if tp_size != 1:
        return False
    try:
        state_keys = set(module.state_dict().keys())
    except Exception:
        return False
    return state_keys.issubset({"weight", "bias"})


def _get_linear_io_features(module: nn.Module) -> tuple[int, int]:
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.in_features), int(module.out_features)
    if hasattr(module, "input_size") and hasattr(module, "output_size"):
        return int(module.input_size), int(module.output_size)
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1]), int(weight.shape[0])
    raise ValueError(f"Cannot infer linear features for module {module.__class__.__name__}")


def _load_linear_weights(src: nn.Module, dst: nn.Module, *, include_bias: bool = True) -> None:
    # Loading via state_dict ensures bitsandbytes hooks see fp weights,
    # and .to("cuda") triggers internal quantization.
    state = src.state_dict()
    keys = {"weight", "bias"} if include_bias else {"weight"}
    filtered = {k: v for k, v in state.items() if k in keys}
    if any(getattr(t, "is_cuda", False) for t in filtered.values()):
        filtered = {k: v.detach().cpu() for k, v in filtered.items()}
    # bitsandbytes 4bit is sensitive to bf16 weights; cast to compute dtype if set.
    compute_dtype = getattr(dst, "compute_dtype", None)
    if compute_dtype is not None:
        filtered = {k: (v.to(compute_dtype) if torch.is_floating_point(v) else v) for k, v in filtered.items()}
    dst.load_state_dict(filtered, strict=False)


def _collect_vllm_linear_meta(module: nn.Module) -> dict[str, object]:
    meta = {}
    for name in ("num_heads", "num_kv_heads", "head_dim", "tp_size", "tp_rank", "input_size", "output_size"):
        if hasattr(module, name):
            meta[name] = getattr(module, name)
    return meta


class _BnbLinearReturnBiasWrapper(nn.Module):
    """Wrapper to preserve (output, bias) semantics for vLLM Linear modules."""

    def __init__(
        self,
        *,
        linear: nn.Module,
        bias: torch.Tensor | None,
        return_bias: bool,
        skip_bias_add: bool,
        meta: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.return_bias = return_bias
        self.skip_bias_add = skip_bias_add
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)
        for key, value in (meta or {}).items():
            setattr(self, key, value)

    def forward(self, x: torch.Tensor):
        out = self.linear(x)
        if not self.return_bias:
            return out
        return out, (self.bias if self.skip_bias_add else None)


def _get_module_device(module: nn.Module) -> torch.device | None:
    try:
        param = next(module.parameters())
        return param.device
    except StopIteration:
        try:
            buf = next(module.buffers())
            return buf.device
        except StopIteration:
            return None


def _set_child_module(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.ModuleList):
        parent[int(name)] = child
        return
    if isinstance(parent, nn.ModuleDict):
        parent[name] = child
        return
    setattr(parent, name, child)
