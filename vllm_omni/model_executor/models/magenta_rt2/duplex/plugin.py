# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-duplex model plugin for Google Magenta RealTime 2 (25 Hz / 48 kHz stereo)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    EncodeAudio,
)
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.common.duplex.pcm_buffer import FixedFramePcmAppendBuffer
from vllm_omni.model_executor.models.magenta_rt2.conditioning import (
    DEFAULT_CFG_SCALES,
    ENCODER_NUM_CHANNELS,
    NUM_DRUM_CHANNELS,
    NUM_MULAN_CHANNELS,
    NUM_PIANOROLL_CHANNELS,
)
from vllm_omni.model_executor.models.magenta_rt2.duplex.capabilities import (
    CHUNK_PERIOD_MS,
    FRAME_SIZE,
    SAMPLE_RATE,
    magenta_rt2_capabilities,
)
from vllm_omni.model_executor.models.magenta_rt2.duplex.data_plane import MagentaRT2DataPlaneSession

if TYPE_CHECKING:
    from vllm.config import ModelConfig

PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "magenta_rt2_internal_state",
    }
)


class MagentaRT2PcmAppendBuffer(FixedFramePcmAppendBuffer):
    """Transactionally frame 48 kHz float PCM into Magenta RT 2 40 ms (1920-sample) ticks."""

    def __init__(self) -> None:
        super().__init__(
            sample_rate_hz=SAMPLE_RATE,
            frame_samples=FRAME_SIZE,
            chunk_period_ms=CHUNK_PERIOD_MS,
            model="MagentaRT2",
        )


@dataclass
class MagentaRT2SessionState(DefaultDuplexModelSessionState):
    """Per-session duplex state with 40 ms PCM framing buffer."""

    audio_buffer: MagentaRT2PcmAppendBuffer = field(default_factory=MagentaRT2PcmAppendBuffer)


def _validate_int_list(value: object, *, expected_len: int, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise DuplexRuntimeConfigError(
            f"{field_name} must be a list of {expected_len} integers",
            code="invalid_conditioning",
        )
    if len(value) != expected_len:
        raise DuplexRuntimeConfigError(
            f"{field_name} must have length {expected_len}, got {len(value)}",
            code="invalid_conditioning",
        )
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise DuplexRuntimeConfigError(f"{field_name} elements must be integers", code="invalid_conditioning") from exc


def _extract_runtime_conditioning(
    config: DuplexSessionConfig,
    base: Mapping[str, object] | None = None,
) -> dict[str, object]:
    out: dict[str, object] = (
        dict(base)
        if base
        else {
            "cfg_scales": dict(DEFAULT_CFG_SCALES),
        }
    )
    extra = config.extra_body if isinstance(config.extra_body, Mapping) else {}

    cfg_raw = extra.get("cfg_scales")
    if isinstance(cfg_raw, Mapping):
        merged_cfg = dict(out.get("cfg_scales", DEFAULT_CFG_SCALES))  # type: ignore[arg-type]
        for key in ("musiccoca", "notes", "drums"):
            if key in cfg_raw and cfg_raw[key] is not None:
                try:
                    merged_cfg[key] = float(cfg_raw[key])  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise DuplexRuntimeConfigError(
                        f"cfg_scales[{key!r}] must be a float",
                        code="invalid_conditioning",
                    ) from exc
        out["cfg_scales"] = merged_cfg

    for key, length in (
        ("mulan_tokens", NUM_MULAN_CHANNELS),
        ("pianoroll_tokens", NUM_PIANOROLL_CHANNELS),
        ("drum_tokens", NUM_DRUM_CHANNELS),
        ("raw_conditioning_tokens", ENCODER_NUM_CHANNELS),
    ):
        if key in extra:
            parsed = _validate_int_list(extra[key], expected_len=length, field_name=key)
            if parsed is not None:
                out[key] = parsed

    if config.temperature is not None:
        out["temperature"] = float(config.temperature)
    elif "temperature" in extra and extra["temperature"] is not None:
        out["temperature"] = float(extra["temperature"])  # type: ignore[arg-type]

    if "top_k" in extra and extra["top_k"] is not None:
        top_k = int(extra["top_k"])  # type: ignore[arg-type]
        if top_k < 0:
            raise DuplexRuntimeConfigError("top_k must be >= 0", code="invalid_sampling")
        out["top_k"] = top_k

    if config.instructions is not None:
        out["instructions"] = str(config.instructions)

    return out


class MagentaRT2DuplexPlugin(DuplexModelPlugin):
    """Magenta RealTime 2 engine policy, live conditioning updates, and stereo data plane."""

    plugin_id = "magenta_realtime"
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS
    silence_continuation_samples = FRAME_SIZE
    silence_continuation_sample_rate_hz = SAMPLE_RATE

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = MagentaRT2DataPlaneSession(encode_audio)

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        if not defaults:
            return defaults
        configured = list(defaults)
        if len(configured) > 1 and isinstance(configured[-1], SamplingParams):
            final = configured[-1].clone()
            final.output_kind = RequestOutputKind.DELTA
            configured[-1] = final
        stage0 = defaults[0]
        if isinstance(stage0, SamplingParams):
            stage0 = stage0.clone()
            stage0.max_tokens = 1
            if isinstance(runtime_config.get("temperature"), (int, float)):
                stage0.temperature = float(runtime_config["temperature"])  # type: ignore[arg-type]
            if isinstance(runtime_config.get("top_k"), int) and int(runtime_config["top_k"]) > 0:  # type: ignore[arg-type]
                stage0.top_k = int(runtime_config["top_k"])  # type: ignore[arg-type]
            configured[0] = stage0
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del session_config, sampling_params, turn_seq, final
        decode_pcm_f32le_payload(
            payload,
            sample_rate_hz=SAMPLE_RATE,
            exact_samples=FRAME_SIZE,
            model="MagentaRT2",
        )
        normalized_payload = dict(payload)  # type: ignore[arg-type]
        duplex: dict[str, Any] = {
            "data_plane": True,
            "session_id": fence.session_id,
            "epoch": fence.epoch,
            "seq": seq,
            "payload": normalized_payload,
            # Pass live conditioning snapshot on each tick so style/note updates via
            # session.update take effect immediately on the next 40 ms frame.
            "runtime_config": dict(runtime_config),
        }
        return DuplexAppendPlan(
            prompt={
                "prompt_token_ids": [0],
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "global_request_id": [fence.session_id],
                    "duplex": duplex,
                },
            }
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        del stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output
        return None

    def create_session_state(self) -> MagentaRT2SessionState:
        return MagentaRT2SessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return magenta_rt2_capabilities(max_sessions=max_sessions)

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        del model_config
        self.validate_client_extra_body(config.extra_body)
        return _extract_runtime_conditioning(config)

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        return _extract_runtime_conditioning(config, base=current)


__all__ = [
    "PRIVATE_RUNTIME_CONFIG_KEYS",
    "MagentaRT2DuplexPlugin",
    "MagentaRT2PcmAppendBuffer",
    "MagentaRT2SessionState",
]
