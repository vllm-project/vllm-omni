# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""In-process codec for the single-stage MOSS-TTS-Local pipeline.

The talker stage owns a ``MossTTSCodecDecoder`` configured as the two-stage
codec stage would be: the same streaming state pool, CUDA graphs and kernel
options, read from ``additional_config["moss_fused_codec"]`` instead of the
connector. It compiles under its own model tag, so its compile cache never
mixes with the talker's.
"""

from __future__ import annotations

import copy
from typing import Any

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

FUSED_CODEC_CONFIG_KEY = "moss_fused_codec"
_DEFAULT_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
_COMPILE_TAG = "moss_fused_codec"


def fused_codec_enabled(vllm_config: VllmConfig) -> bool:
    """The talker is the final audio stage: it must decode its own codes."""
    model_config = vllm_config.model_config
    return getattr(model_config, "engine_output_type", None) == "audio" and bool(
        getattr(model_config, "final_output", False)
    )


def fused_early_first_audio_enabled(vllm_config: VllmConfig) -> bool:
    """Send each stream's first frame through the first-audio sender (see ``fused_audio``)."""
    if not fused_codec_enabled(vllm_config):
        return False
    return bool(int(fused_codec_options(vllm_config).get("fused_early_first_audio", 0)))


def fused_codec_options(vllm_config: VllmConfig) -> dict[str, Any]:
    additional = getattr(vllm_config, "additional_config", None) or {}
    options = additional.get(FUSED_CODEC_CONFIG_KEY) or {}
    if not isinstance(options, dict):
        raise ValueError(f"additional_config.{FUSED_CODEC_CONFIG_KEY} must be a mapping")
    return dict(options)


def _codec_vllm_config(vllm_config: VllmConfig, options: dict[str, Any]) -> VllmConfig:
    """The talker's config as the codec stage would see it."""
    if not getattr(vllm_config.model_config, "async_chunk", False):
        raise ValueError("The single-stage MOSS pipeline requires async_chunk=True")
    if not getattr(vllm_config.model_config, "use_v2_model_runner", False):
        raise ValueError("The single-stage MOSS pipeline requires the MRV2 model runner")
    extra = {key: value for key, value in options.items() if key != "cudagraph_capture_sizes"}
    capacity = int(vllm_config.scheduler_config.max_num_seqs)
    capture_sizes = sorted(
        {int(size) for size in options.get("cudagraph_capture_sizes", _DEFAULT_CAPTURE_SIZES) if 0 < int(size)}
    )
    config = copy.copy(vllm_config)
    config.model_config = copy.copy(vllm_config.model_config)
    # The codec reads its chunking and kernel options as connector extras.
    config.model_config.stage_connector_config = {"extra": extra}
    compilation = copy.copy(vllm_config.compilation_config)
    compilation.cudagraph_capture_sizes = [size for size in capture_sizes if size <= capacity]
    # Separate compile bookkeeping: these fields are filled during compilation.
    compilation.cache_dir = ""
    compilation.local_cache_dir = ""
    compilation.traced_files = set()
    compilation.static_forward_context = {}
    inductor = dict(getattr(compilation, "inductor_compile_config", None) or {})
    inductor.update({"combo_kernels": False, "benchmark_combo_kernel": False})
    compilation.inductor_compile_config = inductor
    config.compilation_config = compilation
    return config


def build_fused_codec(vllm_config: VllmConfig):
    """Load the codec and capture its streaming graphs for the talker stage."""
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder

    options = fused_codec_options(vllm_config)
    codec_config = _codec_vllm_config(vllm_config, options)
    with set_model_tag(_COMPILE_TAG):
        codec = MossTTSCodecDecoder(vllm_config=codec_config)
        codec.load_weights(iter(()))
    session = codec._ensure_stream_session()
    if session is None:
        raise RuntimeError("The single-stage MOSS pipeline requires the streaming codec session")
    if not codec._gpu_stream_output:
        raise RuntimeError("The single-stage MOSS pipeline requires GPU codec stream output")
    logger.info(
        "MOSS single-stage codec ready: capacity=%d chunks=%d/%d graphs=%s packed=%s",
        codec._stream_state_capacity,
        codec._initial_stream_chunk_frames,
        codec._stream_chunk_frames,
        session._cudagraph_wrapper.capture_sizes if session._cudagraph_wrapper is not None else [],
        codec.packed_decode_ready,
    )
    return codec
