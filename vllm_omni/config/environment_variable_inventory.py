# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reviewed inventory of environment variables used by vLLM-Omni.

This module classifies environment-variable names; it does not read them or
make every classified name part of the public configuration contract. In
particular, model-specific variables remain transitional until their recorded
disposition is implemented.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class EnvironmentVariableCategory(str, Enum):
    """Ownership and documentation boundary for an environment variable."""

    PUBLIC_OMNI = "public_omni"
    INHERITED_VLLM = "inherited_vllm"
    PLATFORM_EXTERNAL = "platform_external"
    MODEL_SPECIFIC = "model_specific"
    BENCHMARK_TRANSITIONAL = "benchmark_transitional"
    INTERNAL = "internal"


class ModelEnvironmentVariableDisposition(str, Enum):
    """Required destination for a model-specific environment variable."""

    PROMOTE = "promote"
    REQUEST_SCOPE = "request_scope"
    EXTERNAL = "external"
    INTERNALIZE = "internalize"
    DEPRECATE_REMOVE = "deprecate_remove"


@dataclass(frozen=True)
class EnvironmentVariableClassification:
    """Classification metadata for one statically identifiable name."""

    name: str
    category: EnvironmentVariableCategory
    model_disposition: ModelEnvironmentVariableDisposition | None = None
    redact_value: bool = False

    @property
    def is_public_omni(self) -> bool:
        return self.category is EnvironmentVariableCategory.PUBLIC_OMNI


# New entries must use the VLLM_OMNI_ prefix. The regression test explicitly
# grandfathers the older public names below.
_PUBLIC_OMNI = (
    "DIFFUSION_ATTENTION_BACKEND",
    "DIFFUSION_CACHE_ADAPTER",
    "DIFFUSION_CACHE_BACKEND",
    "OMNI_DIFFUSION_PROMPT_EMBED_CACHE",
    "OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE",
    "OMNI_DIFFUSION_SESSION_STATE_MANAGER",
    "OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS",
    "SPEAKER_MAX_UPLOADED",
    "SPEAKER_SAMPLES_DIR",
    "VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT",
    "VLLM_OMNI_EVENT_DRIVEN_ORCH",
    "VLLM_OMNI_INPUT_WAIT_TIMEOUT_S",
    "VLLM_OMNI_ORCH_MONITOR_PATH",
    "VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY",
    "VLLM_OMNI_SERVER_STORAGE__FILE_TTL",
    "VLLM_OMNI_SERVER_STORAGE__PATH",
    "VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL",
    "VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP",
    "VLLM_OMNI_STORAGE_MAX_CONCURRENCY",
    "VLLM_OMNI_STORAGE_PATH",
    "VLLM_OMNI_USE_QUACK_FP8",
    "VLLM_OMNI_VIDEO_SYNC_TIMEOUT",
    "VLLM_VIDEO_ASYNC_CHUNK",
    "VLLM_VIDEO_AUDIO_DELTA_MODE",
)

_INHERITED_VLLM = (
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "NO_COLOR",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_BATCH_INVARIANT",
    "VLLM_CACHE_ROOT",
    "VLLM_DISABLE_LOG_LOGO",
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
    "VLLM_LOGGING_COLOR",
    "VLLM_MOONCAKE_BOOTSTRAP_PORT",
    "VLLM_PLUGINS",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_RMSNORM",
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_USE_MODELSCOPE",
    "VLLM_USE_OINK_OPS",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_XPU_USE_SAMPLER_KERNEL",
)

_PLATFORM_EXTERNAL = (
    "ASCEND_LAUNCH_BLOCKING",
    "ASCEND_RT_VISIBLE_DEVICES",
    "FLASHINFER_DISABLE_VERSION_CHECK",
    "HF_HOME",
    "HF_HUB_DISABLE_XET",
    "HF_TOKEN",
    "HIP_VISIBLE_DEVICES",
    "HUGGINGFACE_HUB_TOKEN",
    "MASTER_ADDR",
    "MASTER_PORT",
    "MINDIE_SD_FA_TYPE",
    "MORI_RDMA_DEVICES",
    "MUSA_VISIBLE_DEVICES",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "NCCL_ASYNC_ERROR_HANDLING",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OPENAI_API_KEY",
    "PYTHONPATH",
    "QUACK_CACHE_DIR",
    "RANK",
    "RANK_ID",
    "RAY_RAYLET_PID",
    "RDMA_DEVICE_NAME",
    "SAGE_ATTN_TENSOR_LAYOUT",
    "VLLM_LOCAL_RANK",
    "WORLD_SIZE",
    "XDG_CACHE_HOME",
)

_MODEL_PROMOTE = (
    "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH",
    "COSMOS3_SOUND_TOKENIZER_CONFIG_PATH",
    "COSMOS3_SOUND_TOKENIZER_PATH",
    "COSYVOICE3_ESTIMATOR_ONNX",
    "COSYVOICE3_TRT",
    "COSYVOICE3_TRT_CACHE",
    "DYNIN_CONFIG_PATH",
    "DYNIN_MAGVIT_REMOTE_CODE_LOCAL_FILES_ONLY",
    "DYNIN_MAGVIT_REMOTE_CODE_REPO_ID",
    "DYNIN_MAGVIT_REMOTE_CODE_REVISION",
    "DYNIN_REMOTE_CODE_LOCAL_FILES_ONLY",
    "DYNIN_REMOTE_CODE_REPO_ID",
    "DYNIN_REMOTE_CODE_REVISION",
    "HIGGS_AUDIO_TOKENIZER_PATH",
    "INTERNVLA_A1_COSMOS_DECODER_PATH",
    "INTERNVLA_A1_COSMOS_DIR",
    "INTERNVLA_A1_COSMOS_ENCODER_PATH",
    "INTERNVLA_A1_PROCESSOR_DIR",
    "MIMO_AUDIO_TOKENIZER_CUDA_GRAPH",
    "MIMO_AUDIO_TOKENIZER_DEVICE",
    "MIMO_AUDIO_TOKENIZER_PATH",
    "MING_CFM_CUDAGRAPH",
    "MINICPMO_TOKEN2WAV_TRT",
    "MINICPMO_TOKEN2WAV_TRT_CACHE",
    "MINICPMO_TOKEN2WAV_TRT_DTYPE",
    "NEMOTRON_VOICECHAT_LLM_PATH",
    "OMNIVOICE_CUDA_GRAPH",
    "OMNIVOICE_TF32",
    "VLLM_GEPARD_CHUNK_FRAMES",
    "VLLM_GEPARD_FIRST_CHUNK_FRAMES",
    "VLLM_GEPARD_LOOKBACK_FRAMES",
    "VLLM_OMNI_FISH_KVCACHE_ATTN",
    "VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER",
)

_MODEL_REQUEST_SCOPE = (
    "DOTS_TTS_DIT_NUM_STEPS",
    "GR00T_NOISE_SEED",
    "STEP_AUDIO2_DEFAULT_PROMPT_WAV",
    "VLLM_GEPARD_END_FADE_MS",
    "VLLM_GEPARD_END_SILENCE_MS",
    "VLLM_GEPARD_GREEDY",
)

# No audited model variable is currently owned by a supported third-party
# contract. Keep the disposition explicit so future reviews cannot silently
# fold an externally owned name into "promote".
_MODEL_EXTERNAL: tuple[str, ...] = ()

_MODEL_INTERNALIZE = (
    "DOTS_TTS_BETA_TRACE",
    "DYNIN_S2U_VENDOR_ROOT",
    "DZ_PHASE_TIMING",
    "MING_TTS_STAGE1_FINAL_LOG",
    "MOSS_TTS_DEBUG_STOP",
    "NEMOTRON_VOICECHAT_DEBUG_FUNCTION_TIMELINE",
    "VLLM_OMNI_QWEN3_CODE2WAV_BATCH_STATS",
    "VLLM_OMNI_QWEN3_CODE2WAV_BATCH_STATS_LOG_EVERY",
    "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS",
    "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_FILE",
    "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_LOG_EVERY",
)

_MODEL_DEPRECATE_REMOVE = (
    "HIGGS_AUDIO_V2_TOKENIZER_PATH",
    "VLLM_OMNI_LINGBOT_ACTION_ROOT",
    "VLLM_OMNI_VOXCPM_CODE_PATH",
    "VLLM_USE_FLASHINFER_MOE_FP16",
    "XCODEC1_PATH",
    "model_stage",
)

_BENCHMARK_TRANSITIONAL = (
    "DAILY_OMNI_EXTRACT_MODE",
    "DAILY_OMNI_SAVE_EVAL_ITEMS",
    "MAX_NUM_FRAMES",
    "SEED_TTS_EVAL_DEVICE",
    "SEED_TTS_HF_WHISPER_MODEL",
    "SEED_TTS_SIM_DEVICE",
    "SEED_TTS_SIM_EVAL",
    "SEED_TTS_UTMOS_EVAL",
    "SEED_TTS_UTMOS_HF_REPO",
    "SEED_TTS_UTMOS_JIT_FILE",
    "SEED_TTS_WAVLM_MAX_SECONDS",
    "SEED_TTS_WAVLM_MIN_SAMPLES",
    "SEED_TTS_WAVLM_MODEL",
    "SEED_TTS_WER_EVAL",
    "SEED_TTS_WER_SAVE_AUDIO_DIR",
    "SEED_TTS_WER_SAVE_ITEMS",
    "VLLM_DAILY_OMNI_MEDIA_REPO",
    "VLLM_OMNI_BENCH_AUDIO_CHANNELS",
    "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S",
    "VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE",
)

_INTERNAL = (
    "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT",
    "VLLM_OMNI_REPLICA_ID",
)

_REDACTED = frozenset(
    {
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "OPENAI_API_KEY",
    }
)


def _build_inventory() -> Mapping[str, EnvironmentVariableClassification]:
    inventory: dict[str, EnvironmentVariableClassification] = {}

    def add(
        names: tuple[str, ...],
        category: EnvironmentVariableCategory,
        disposition: ModelEnvironmentVariableDisposition | None = None,
    ) -> None:
        for name in names:
            if name in inventory:
                raise ValueError(f"Environment variable {name!r} is classified more than once")
            inventory[name] = EnvironmentVariableClassification(
                name=name,
                category=category,
                model_disposition=disposition,
                redact_value=name in _REDACTED,
            )

    add(_PUBLIC_OMNI, EnvironmentVariableCategory.PUBLIC_OMNI)
    add(_INHERITED_VLLM, EnvironmentVariableCategory.INHERITED_VLLM)
    add(_PLATFORM_EXTERNAL, EnvironmentVariableCategory.PLATFORM_EXTERNAL)
    add(_MODEL_PROMOTE, EnvironmentVariableCategory.MODEL_SPECIFIC, ModelEnvironmentVariableDisposition.PROMOTE)
    add(
        _MODEL_REQUEST_SCOPE,
        EnvironmentVariableCategory.MODEL_SPECIFIC,
        ModelEnvironmentVariableDisposition.REQUEST_SCOPE,
    )
    add(_MODEL_EXTERNAL, EnvironmentVariableCategory.MODEL_SPECIFIC, ModelEnvironmentVariableDisposition.EXTERNAL)
    add(
        _MODEL_INTERNALIZE,
        EnvironmentVariableCategory.MODEL_SPECIFIC,
        ModelEnvironmentVariableDisposition.INTERNALIZE,
    )
    add(
        _MODEL_DEPRECATE_REMOVE,
        EnvironmentVariableCategory.MODEL_SPECIFIC,
        ModelEnvironmentVariableDisposition.DEPRECATE_REMOVE,
    )
    add(_BENCHMARK_TRANSITIONAL, EnvironmentVariableCategory.BENCHMARK_TRANSITIONAL)
    add(_INTERNAL, EnvironmentVariableCategory.INTERNAL)
    return MappingProxyType(dict(sorted(inventory.items())))


# All reviewed environment-variable names, indexed by exact name. This is
# classification metadata, not the executable value registry used by
# ``vllm.envs``.
ENVIRONMENT_VARIABLE_INVENTORY: Mapping[str, EnvironmentVariableClassification] = _build_inventory()
