# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration shared by the Talker's first-frame decoder and the Code2Wav stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig

_CODEC_KERNEL_DEFAULTS = {
    "codec_cudnn_fp32_fallback": True,
    "codec_cudnn_benchmark": False,
    "codec_fused_snake": False,
}


def codec_kernel_options(extra: Mapping[str, Any]) -> dict[str, bool]:
    return {key: extra.get(key, default) for key, default in _CODEC_KERNEL_DEFAULTS.items()}


def first_frame_decoders_match(
    talker: BaseVllmOmniStageConfig,
    codec: BaseVllmOmniStageConfig,
    talker_spec: Mapping[str, Any],
    codec_spec: Mapping[str, Any],
) -> bool:
    talker_model, codec_model = talker.model_config, codec.model_config
    talker_extra, codec_extra = talker_spec.get("extra", {}), codec_spec.get("extra", {})
    devices = talker.runtime_config.devices
    return (
        # The cuDNN precision workaround depends on the GPU architecture.
        devices is not None
        and devices.isdecimal()
        and devices == codec.runtime_config.devices
        and talker.weight_transfer_config is None
        and codec.weight_transfer_config is None
        and talker_model.model == codec_model.model
        and talker_model.model_subdir == codec_model.model_subdir
        and talker_model.model_arch == codec_model.model_arch
        and talker_model.revision == codec_model.revision
        and talker_model.code_revision == codec_model.code_revision
        and talker_model.trust_remote_code == codec_model.trust_remote_code
        and talker_model.dtype == codec_model.dtype
        and talker.quantization_config == codec.quantization_config
        and talker_model.hf_config_path == codec_model.hf_config_path
        and not callable(talker_model.hf_overrides)
        and (talker_model.hf_overrides or {}) == (codec_model.hf_overrides or {})
        and talker.load_config.config_format == codec.load_config.config_format
        and talker.load_config.load_format == codec.load_config.load_format != "dummy"
        and talker.load_config.model_loader_extra_config == codec.load_config.model_loader_extra_config
        and codec_kernel_options(talker_extra) == codec_kernel_options(codec_extra)
    )
