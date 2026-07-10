# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn

from vllm_omni.quantization.fp8_blockwise_w8a16 import (
    fp8_w8a16_selected,
    is_fp8_blockwise_w8a16_config,
)
from vllm_omni.quantization.nvfp4_blockwise import is_nvfp4_blockwise_w4a16_config

from .modelopt import (
    ModelOptFp8CheckpointAdapter,
    ModelOptMixedPrecisionCheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
)
from .modelopt_native import CheckpointIntegrityError, ModelOptNativeFp8CheckpointAdapter
from .modelopt_native_fp8_w8a16 import ModelOptNativeFp8W8A16CheckpointAdapter
from .modelopt_native_nvfp4 import (
    CheckpointIntegrityError as Nvfp4CheckpointIntegrityError,
)
from .modelopt_native_nvfp4 import ModelOptNativeNvfp4CheckpointAdapter

# Recipe-gated dispatch: the FP8-blockwise resident W8A16 path is explicit
# opt-in via ``VLLM_OMNI_FP8_BLOCKWISE_W8A16=1`` and still requires the root
# ``quantization_config.json`` recipe. Load-time dequant remains the default.


def _model_dtype(model: nn.Module) -> torch.dtype:
    param = next(model.parameters(), None)
    return param.dtype if param is not None else torch.bfloat16


def get_checkpoint_adapter(
    model: nn.Module,
    source: object,
    quant_config: object | None,
    use_safetensors: bool,
) -> (
    ModelOptFp8CheckpointAdapter
    | ModelOptNvFp4CheckpointAdapter
    | ModelOptMixedPrecisionCheckpointAdapter
    | ModelOptNativeFp8CheckpointAdapter
    | ModelOptNativeFp8W8A16CheckpointAdapter
    | ModelOptNativeNvfp4CheckpointAdapter
    | None
):
    if use_safetensors:
        # Checkpoint-driven sidecar detection. The load-time FP8 dequant path is
        # independent of quant_config because it emits full-precision weights.
        # Resident native adapters must match the active config, otherwise the
        # loaded tensor names/dtypes can disagree with the constructed layers.
        #
        # The optional FP8 W8A16 path must precede the dequant FP8 adapter
        # because both key off the same sidecar. The predicate is false unless
        # the explicit opt-in is set and the root recipe matches, so dequant is
        # the default and NVFP4 is unaffected.
        if fp8_w8a16_selected(source):
            if not is_fp8_blockwise_w8a16_config(quant_config):
                raise CheckpointIntegrityError(
                    "FP8 W8A16 resident checkpoint sidecar detected, but the active quant_config is not "
                    "the FP8 blockwise W8A16 config. Refusing to load resident FP8 weights into "
                    "incompatible layers."
                )
            w8a16_adapter = ModelOptNativeFp8W8A16CheckpointAdapter.detect(source, target_dtype=_model_dtype(model))
            if w8a16_adapter is not None:
                return w8a16_adapter
        native_adapter = ModelOptNativeFp8CheckpointAdapter.detect(source, target_dtype=_model_dtype(model))
        if native_adapter is not None:
            return native_adapter
        nvfp4_native_adapter = ModelOptNativeNvfp4CheckpointAdapter.detect(source, target_dtype=_model_dtype(model))
        if nvfp4_native_adapter is not None:
            if not is_nvfp4_blockwise_w4a16_config(quant_config):
                raise Nvfp4CheckpointIntegrityError(
                    "NVFP4 native checkpoint sidecar detected, but the active quant_config is not "
                    "the NVFP4 blockwise W4A16 config. Refusing to load resident NVFP4 weights into "
                    "incompatible layers."
                )
            return nvfp4_native_adapter
    if ModelOptFp8CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptFp8CheckpointAdapter(model, source)
    if ModelOptNvFp4CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptNvFp4CheckpointAdapter(model, source)
    if ModelOptMixedPrecisionCheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptMixedPrecisionCheckpointAdapter(model, source)
    return None


__all__ = [
    "ModelOptFp8CheckpointAdapter",
    "ModelOptMixedPrecisionCheckpointAdapter",
    "ModelOptNativeFp8CheckpointAdapter",
    "ModelOptNativeFp8W8A16CheckpointAdapter",
    "ModelOptNativeNvfp4CheckpointAdapter",
    "ModelOptNvFp4CheckpointAdapter",
    "get_checkpoint_adapter",
]
