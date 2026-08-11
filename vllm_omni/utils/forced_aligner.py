# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forced-aligner config + word-timestamp decoding for TTS.

The aligner runs as a pooling stage appended to the pipeline by
``--forced-aligner`` (see :func:`inject_forced_aligner_stage`). Qwen-specific
word segmentation / prompt building / marker repair live in
:mod:`vllm_omni.utils.qwen3_force_align_processor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from vllm_omni.utils import qwen3_force_align_processor as _processor

if TYPE_CHECKING:
    from vllm_omni.config.stage_config import DeployConfig, PipelineConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "deploy" / "qwen3_tts_forced_aligner.yaml"


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Internal alignment record. Serialized to a plain JSON object
    (``{"word", "start_ms", "end_ms"}``) at the WebSocket boundary.
    """

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class ForcedAlignerConfig:
    """Plain-data config for the forced-aligner pipeline stage; fields are
    consumed by :func:`inject_forced_aligner_stage`."""

    model: str
    runner: str | None = None
    architecture: str | None = None
    pooling_task: str | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    max_model_len: int | None = None
    trust_remote_code: bool | None = None


def _load_forced_aligner_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get("forced_aligner", raw))


def build_forced_aligner_config(model: str | None, config_path: str | None = None) -> ForcedAlignerConfig | None:
    """Build a config from the CLI values, or ``None`` when the feature is off.

    Precedence (lowest to highest): the packaged default YAML
    (:data:`_DEFAULT_CONFIG_PATH`, Qwen deploy defaults) -> a user YAML passed
    via ``--forced-aligner-config`` -> the ``--forced-aligner`` model path. The
    feature is off (returns ``None``) unless a model resolves from this chain.
    Per-field overrides such as ``gpu_memory_utilization`` live in the YAML.
    """
    config_data: dict[str, Any] = {}
    if config_path or model:
        config_data.update(_load_forced_aligner_yaml(_DEFAULT_CONFIG_PATH))
    if config_path:
        config_data.update(_load_forced_aligner_yaml(config_path))
    if model:
        config_data["model"] = str(model)
    if not config_data.get("model"):
        return None
    allowed = set(ForcedAlignerConfig.__dataclass_fields__)
    return ForcedAlignerConfig(**{k: v for k, v in config_data.items() if k in allowed})


def inject_forced_aligner_stage(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    cli_overrides: dict[str, Any],
) -> tuple[PipelineConfig, DeployConfig]:
    """Append a forced-aligner pooling stage to the pipeline tail when
    ``--forced-aligner`` or ``--forced-aligner-config`` resolves an aligner
    model; no-op otherwise. The stage runs the aligner model with
    ``runner="pooling"``, consuming the previous stage's audio and emitting a
    terminal word-timestamps side-output.
    """
    if not (cli_overrides.get("forced_aligner") or cli_overrides.get("forced_aligner_config")):
        return pipeline, deploy

    from vllm_omni.config.stage_config import (
        StageDeployConfig,
        StageExecutionType,
        StagePipelineConfig,
    )
    from vllm_omni.model_executor.stage_input_processors.forced_aligner import (
        POOLING_OUTPUT_DECODER_PATH,
        TIMESTAMPS_MODALITY,
    )

    fa = build_forced_aligner_config(
        cli_overrides.get("forced_aligner"),
        cli_overrides.get("forced_aligner_config"),
    )
    if fa is None:
        return pipeline, deploy

    _FA = "vllm_omni.model_executor.stage_input_processors.forced_aligner"
    new_id = len(pipeline.stages)
    aligner_ps = StagePipelineConfig(
        stage_id=new_id,
        model_stage="forced_aligner",
        # Pooling stage = LLM_AR stage run with runner="pooling" (vLLM derives
        # is_pooling_model from runner_type).
        execution_type=StageExecutionType.LLM_AR,
        input_sources=(new_id - 1,),
        final_output=True,
        final_output_type=TIMESTAMPS_MODALITY,
        owns_tokenizer=True,
        requires_multimodal_data=True,
        model_arch=fa.architecture,
        custom_process_input_func=f"{_FA}.code2wav2aligner",
    )
    extended = replace(pipeline, stages=pipeline.stages + (aligner_ps,))

    engine_extras: dict[str, Any] = {
        "model": fa.model,
        "runner": fa.runner or "pooling",
        # Worker-side decoder hook: pooler logits -> word-timestamps tensor.
        "pooling_output_decoder": POOLING_OUTPUT_DECODER_PATH,
    }
    if fa.architecture:
        engine_extras["hf_overrides"] = {"architectures": [fa.architecture]}
    if fa.trust_remote_code is not None:
        engine_extras["trust_remote_code"] = fa.trust_remote_code
    if fa.dtype is not None:
        engine_extras["dtype"] = fa.dtype
    aligner_ds = StageDeployConfig(
        stage_id=new_id,
        devices=cli_overrides.get("forced_aligner_device"),
        gpu_memory_utilization=fa.gpu_memory_utilization,
        max_model_len=fa.max_model_len,
        engine_extras=engine_extras,
        default_pooling_params={"task": fa.pooling_task or "token_classify"},
        # Pooling is one-shot -> sync AR scheduler (the async one assumes
        # token-by-token generation).
        async_scheduling=False,
    )
    deploy = replace(deploy, stages=list(deploy.stages) + [aligner_ds])
    logger.info("[stage_init] Injected forced-aligner stage %d (model=%s)", new_id, fa.model)
    return extended, deploy


def _decode_timestamps(
    *,
    logits: Any,
    words: list[str],
    timestamp_positions: list[int],
    classify_num: int,
    audio_duration_ms: float,
    timestamp_segment_time_ms: float | None = None,
) -> list[WordTimestamp]:
    """Translate ``[n_token, classify_num]`` logits into word timestamps.

    ``words`` must be the exact segmentation used to build the prompt (see
    :func:`qwen3_force_align_processor.segment_words`); each word owns two
    consecutive markers.
    """
    arr = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits [n_token, classify_num]; got shape {arr.shape}")
    if arr.shape[1] != classify_num:
        raise ValueError(
            f"Logits last dim {arr.shape[1]} != classify_num {classify_num}; "
            "model config and prompt template may be out of sync."
        )

    expected = len(words) * 2
    if len(timestamp_positions) != expected:
        logger.warning(
            "Got %d timestamp positions but text has %d words (expected %d start/end markers); "
            "returning empty alignment.",
            len(timestamp_positions),
            len(words),
            expected,
        )
        return []

    marker_logits = arr[timestamp_positions, :]
    bin_idx = marker_logits.argmax(axis=-1)
    bin_size_ms = (
        float(timestamp_segment_time_ms)
        if timestamp_segment_time_ms is not None
        else (audio_duration_ms / classify_num if classify_num > 0 else 0.0)
    )

    # Repair non-monotonic bins across the whole start/end sequence before
    # pairing, so each (start, end) pair stays ordered and one bad bin can't
    # corrupt its neighbours.
    marker_ms = _processor.fix_timestamp([int(round(int(b) * bin_size_ms)) for b in bin_idx])

    # Pair markers into per-word intervals, enforcing the wire contract here so
    # callers don't have to: monotonic and non-overlapping across words, with
    # start <= end and nothing past the audio length.
    duration_ms = int(round(audio_duration_ms))
    out: list[WordTimestamp] = []
    prev_end_ms = 0
    for i, word in enumerate(words):
        start_ms = min(max(marker_ms[i * 2], prev_end_ms), duration_ms)
        end_ms = min(max(marker_ms[i * 2 + 1], start_ms), duration_ms)
        out.append(WordTimestamp(word=word, start_ms=start_ms, end_ms=end_ms))
        prev_end_ms = end_ms
    return out


def extract_word_timestamps(res: Any, text: str, language: str | None = None) -> list[dict] | None:
    """Read a forced-aligner stage's engine output into per-word timestamps.

    Returns ``[{word, start_ms, end_ms}, ...]`` from the stage's pooling output
    (``res.outputs.data`` is an int32 ``[n_words, 2]`` tensor). Words come from
    re-segmenting ``text``; any mismatch or non-aligner input returns ``None``.
    """
    if not text:
        return None
    try:
        outs = getattr(res, "outputs", None)
        if outs is None:
            return None
        # The aligner stage's res.outputs is a single PoolingOutput (not a list).
        # Its .data is a [n_words, 2] int32 tensor of [start_ms, end_ms] per word.
        out0 = outs[0] if isinstance(outs, (list, tuple)) else outs
        ms = getattr(out0, "data", None)
        if ms is None or not hasattr(ms, "tolist"):
            return None
        ms_list = ms.tolist()
        if not ms_list or not isinstance(ms_list[0], (list, tuple)):
            return None

        words = _processor.segment_words(text, language)
        if len(words) != len(ms_list):
            # Segmentation drifted from the engine side; pairing positionally would mislabel every word.
            logger.warning(
                "word/timestamp count mismatch (%d words vs %d intervals); returning no timestamps",
                len(words),
                len(ms_list),
            )
            return None

        return [{"word": w, "start_ms": int(p[0]), "end_ms": int(p[1])} for w, p in zip(words, ms_list)]
    except Exception:
        logger.debug("failed to extract word timestamps from aligner output", exc_info=True)
        return None
