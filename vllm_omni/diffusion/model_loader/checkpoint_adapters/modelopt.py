# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping

logger = init_logger(__name__)

MODEL_OPT_SCALE_SUFFIXES = (".input_scale", ".weight_scale", ".weight_scale_inv")
DEFAULT_PACKED_MODULES_MAPPING = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "w13": ("w1", "w3"),
}
FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


@dataclass
class _AdaptState:
    scale_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_weights: dict[str, list[tuple[str, torch.Tensor, torch.dtype]]] = field(default_factory=dict)
    skipped_scales: int = 0
    dequantized_weights: int = 0


class ModelOptFp8CheckpointAdapter:
    def __init__(self, model: nn.Module, source: object):
        self._loadable_tensors = self._get_model_loadable_tensors(model)
        self._weights_mapper = self._get_weights_mapper(model)
        self._source_label = getattr(source, "prefix", "") or getattr(source, "subfolder", None) or "model"

    @classmethod
    def is_compatible(
        cls,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        return use_safetensors and cls._is_transformer_source(source) and cls._is_checkpoint_quant_config(quant_config)

    @staticmethod
    def _is_transformer_source(source: object) -> bool:
        if getattr(source, "subfolder", None) == "transformer":
            return True
        return str(getattr(source, "prefix", "")).startswith("transformer.")

    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        return (
            quant_config is not None
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "modelopt"
            and bool(getattr(quant_config, "is_checkpoint_fp8_serialized", False))
        )

    @staticmethod
    def _get_model_loadable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
        loadable_tensors: dict[str, torch.Tensor] = {name: param for name, param in model.named_parameters()}
        loadable_tensors.update({name: buffer for name, buffer in model.named_buffers()})
        return loadable_tensors

    @staticmethod
    def _is_scale(name: str) -> bool:
        return name.endswith(MODEL_OPT_SCALE_SUFFIXES)

    @staticmethod
    def _is_fp8_tensor(tensor: torch.Tensor) -> bool:
        return tensor.dtype in FP8_DTYPES

    @staticmethod
    def _get_weight_scale_name(weight_name: str) -> str | None:
        if weight_name.endswith(".weight"):
            return weight_name[: -len(".weight")] + ".weight_scale"
        return None

    @classmethod
    def _get_weights_mapper(cls, model: nn.Module) -> WeightsMapper:
        mapping = {
            packed_name: tuple(shard_names) for packed_name, shard_names in DEFAULT_PACKED_MODULES_MAPPING.items()
        }
        mapping.update(
            {
                str(packed_name): tuple(str(shard_name) for shard_name in shard_names)
                for packed_name, shard_names in get_packed_modules_mapping(model).items()
            }
        )

        orig_to_new_substr = {".to_out.0.": ".to_out."}
        orig_to_new_prefix: dict[str, str] = {}
        for packed_name, shard_names in mapping.items():
            for shard_name in shard_names:
                orig_to_new_substr[f".{shard_name}."] = f".{packed_name}."
                orig_to_new_prefix[f"{shard_name}."] = f"{packed_name}."
        return WeightsMapper(
            orig_to_new_substr=orig_to_new_substr,
            orig_to_new_prefix=orig_to_new_prefix,
        )

    def _resolve_target_name(self, name: str) -> str | None:
        if name in self._loadable_tensors:
            return name

        for candidate in self._weights_mapper.apply_list([name]):
            if candidate != name and candidate in self._loadable_tensors:
                return candidate
        return None

    @staticmethod
    def _reshape_weight_scale(scale: torch.Tensor, weight_shape: torch.Size) -> torch.Tensor:
        if scale.numel() == 1:
            return scale.reshape(())
        if len(weight_shape) == 2 and scale.ndim == 1 and scale.shape[0] == weight_shape[0]:
            return scale.reshape(-1, 1)
        if tuple(scale.shape) == tuple(weight_shape):
            return scale
        if (
            len(weight_shape) == 2
            and scale.ndim == 4
            and scale.shape[1] == 1
            and scale.shape[3] == 1
            and weight_shape[0] % scale.shape[0] == 0
            and weight_shape[1] % scale.shape[2] == 0
        ):
            block_n = weight_shape[0] // scale.shape[0]
            block_k = weight_shape[1] // scale.shape[2]
            return scale.expand(scale.shape[0], block_n, scale.shape[2], block_k).reshape(weight_shape)
        raise ValueError(f"Unsupported ModelOpt FP8 weight_scale shape {tuple(scale.shape)} for weight {weight_shape}")

    def _dequantize_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        state: _AdaptState,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_name = self._get_weight_scale_name(name)
        if scale_name is None or scale_name not in state.scale_tensors:
            raise ValueError(f"Missing ModelOpt FP8 weight_scale for full-precision target weight {name!r}")

        scale = state.scale_tensors[scale_name].to(dtype=target_dtype, device=loaded_weight.device)
        scale = self._reshape_weight_scale(scale, loaded_weight.shape)
        return loaded_weight.to(dtype=target_dtype) * scale

    def _flush_pending_weights(
        self,
        scale_name: str,
        state: _AdaptState,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for weight_name, weight_tensor, target_dtype in state.pending_weights.pop(scale_name, []):
            yield weight_name, self._dequantize_weight(weight_name, weight_tensor, state, target_dtype)
            state.dequantized_weights += 1

    def _handle_scale_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        target_name: str | None,
        state: _AdaptState,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        state.scale_tensors[name] = tensor
        if target_name is None:
            state.skipped_scales += 1
        else:
            yield name, tensor
        yield from self._flush_pending_weights(name, state)

    def _target_dtype_for_dequantization(
        self,
        tensor: torch.Tensor,
        target_name: str | None,
    ) -> torch.dtype | None:
        if target_name is None or not self._is_fp8_tensor(tensor):
            return None

        target_dtype = self._loadable_tensors[target_name].dtype
        if target_dtype in FP8_DTYPES:
            return None
        return target_dtype

    def _maybe_dequantize_or_defer_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
        state: _AdaptState,
    ) -> torch.Tensor | None:
        scale_name = self._get_weight_scale_name(name)
        if scale_name is None:
            raise ValueError(f"Missing ModelOpt FP8 weight_scale name for weight {name!r}")

        if scale_name not in state.scale_tensors:
            state.pending_weights.setdefault(scale_name, []).append((name, tensor, target_dtype))
            return None

        state.dequantized_weights += 1
        return self._dequantize_weight(name, tensor, state, target_dtype)

    @staticmethod
    def _check_pending_weights(state: _AdaptState) -> None:
        if not state.pending_weights:
            return

        missing_scale_names = ", ".join(repr(name) for name in sorted(state.pending_weights))
        raise ValueError(f"Missing ModelOpt FP8 weight_scale for full-precision target weights: {missing_scale_names}")

    def _log_adaptation_summary(self, state: _AdaptState) -> None:
        if not state.skipped_scales and not state.dequantized_weights:
            return

        logger.info_once(
            "Adapted ModelOpt FP8 %s weights: dequantized %d full-precision weights, skipped %d scale tensors",
            self._source_label,
            state.dequantized_weights,
            state.skipped_scales,
        )

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        state = _AdaptState()

        for name, tensor in weights:
            target_name = self._resolve_target_name(name)
            if self._is_scale(name):
                yield from self._handle_scale_tensor(name, tensor, target_name, state)
                continue

            target_dtype = self._target_dtype_for_dequantization(tensor, target_name)
            if target_dtype is not None:
                tensor = self._maybe_dequantize_or_defer_weight(name, tensor, target_dtype, state)
                if tensor is None:
                    continue
            yield name, tensor

        self._check_pending_weights(state)
        self._log_adaptation_summary(state)
