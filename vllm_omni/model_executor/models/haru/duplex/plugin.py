# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru_vllm/haru_vllm/duplex/plugin.py @ d3612b0.
"""Full-duplex serving policy: one 133 ms frame per append, one resumable request per conversation.

Session fields: ``instructions`` is the system prompt (the character card) and
``ref_audio`` an audio data URI of the voice to speak with. ``extra_body``
takes ``guidance_scale``, ``top_k``, ``top_p``, ``eop_bias``, ``speak_first``,
``dry`` and ``seed``. They are fixed when the session is created.

The first append of a session starts it: the prompt runs and the model's
frame 0 comes back; the audio of that append is not used. From then on every
appended frame ``k`` is answered by model frame ``k + 1``.
"""

from __future__ import annotations

import asyncio
import binascii
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pybase64 as base64
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
)
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.common.duplex.pcm_buffer import FixedFramePcmAppendBuffer
from vllm_omni.model_executor.models.haru.audio import decode_audio, prepare_reference
from vllm_omni.model_executor.models.haru.configuration_haru import HaruSpec
from vllm_omni.model_executor.models.haru.duplex.data_plane import HaruDataPlane
from vllm_omni.model_executor.models.haru.loaders import checkpoint_file, load_tokenizer
from vllm_omni.model_executor.models.haru.prompt import prompt_length
from vllm_omni.model_executor.models.haru.runtime import SessionOptions

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from vllm.config import ModelConfig

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 3200
# Frames are 133.3 ms; the whole-millisecond period is rounded up so that the
# engine's silence continuation never runs ahead of a client sending in real time.
CHUNK_PERIOD_MS = 134
MAX_REFERENCE_SECONDS = 180.0
_MAX_EOP_BIAS = 100.0

_EXTRA_BODY = {
    "guidance_scale": float,
    "top_k": int,
    "top_p": float,
    "eop_bias": float,
    "speak_first": bool,
    "dry": bool,
    "seed": int,
}


def _frame_buffer() -> FixedFramePcmAppendBuffer:
    return FixedFramePcmAppendBuffer(
        sample_rate_hz=SAMPLE_RATE, frame_samples=FRAME_SAMPLES, chunk_period_ms=CHUNK_PERIOD_MS, model="Haru"
    )


@dataclass
class HaruSessionState(DefaultDuplexModelSessionState):
    audio_buffer: FixedFramePcmAppendBuffer = field(default_factory=_frame_buffer)


def _capabilities(max_sessions: int) -> DuplexCapabilities:
    multi = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=True,
        supports_external_turn_signal=False,
        supports_client_commit=False,
        supports_barge_in=False,
        supports_playback_ack=True,
        supports_input_append=True,
        supports_replace_latest_chunk=False,
        supports_reencode_context=False,
        supports_rollback_to_checkpoint=False,
        supports_turn_commit_only=False,
        supports_model_internal_state=True,
        supports_stage_resumption=True,
        supports_core_resumable_request=True,
        supports_stage_connector_handoff=False,
        supports_independent_io_streams=True,
        supports_realtime_endpoint=True,
        supports_multi_session=multi,
        supports_multi_session_same_replica=multi,
        supports_session_lease=True,
        supports_session_resume=False,
        session_admission_mode="engine_managed",
        supports_audio_truncate=False,
        supports_chat_completions=False,
        requires_model_runner_kv=True,
        requires_native_stage_role=True,
        adapter_patterns=["scheduler_data_plane"],
        signal_sources=["model_native", "client_event"],
        stage_handoff_transport="scheduler_data_plane",
        chunk_period_ms=CHUNK_PERIOD_MS,
        target_barge_in_latency_ms=None,
    )


def _decode_reference(ref_audio: str | None) -> tuple[str, int]:
    """Audio data URI -> (base64 PCM16 at 24 kHz, sample count); silence-trimmed, at most 180 s."""
    if not ref_audio:
        return "", 0
    if not ref_audio.startswith("data:") or ";base64," not in ref_audio:
        raise DuplexRuntimeConfigError("ref_audio must be a base64 audio data URI", code="invalid_ref_audio")
    try:
        audio = decode_audio(base64.b64decode(ref_audio.split(";base64,", 1)[1], validate=True), SAMPLE_RATE)
    except (ValueError, binascii.Error) as error:
        raise DuplexRuntimeConfigError(f"ref_audio could not be decoded: {error}", code="invalid_ref_audio") from error
    audio = prepare_reference(audio, MAX_REFERENCE_SECONDS, SAMPLE_RATE)
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii"), len(pcm)


def _client_options(extra_body: Mapping[str, object] | None) -> dict[str, object]:
    """The typed Haru options a client set in ``extra_body`` (unset and null keys are left out)."""
    extra: dict[str, object] = {}
    for name, kind in _EXTRA_BODY.items():
        value = (extra_body or {}).get(name)
        if value is None:
            continue
        try:
            extra[name] = kind(value)
        except (TypeError, ValueError) as error:
            raise DuplexRuntimeConfigError(f"extra_body.{name} must be {kind.__name__}") from error
    return extra


def _check_ranges(extra: Mapping[str, object], temperature: float | None) -> None:
    """Refuse values that would fail in the worker step (seed overflow, NaN or inf logits)."""
    checks = {
        "seed": lambda v: 0 <= v < 2**63,
        "top_k": lambda v: v >= -1,
        "top_p": lambda v: 0.0 < v <= 1.0,
        "eop_bias": lambda v: math.isfinite(v) and abs(v) <= _MAX_EOP_BIAS,
    }
    for name, valid in checks.items():
        if name in extra and not valid(extra[name]):
            raise DuplexRuntimeConfigError(f"extra_body.{name} is out of range: {extra[name]!r}")
    if temperature is not None and not (math.isfinite(float(temperature)) and float(temperature) >= 0.0):
        raise DuplexRuntimeConfigError(f"temperature is out of range: {temperature!r}")


def _load_checkpoint(model_path: str) -> tuple[Tokenizer, HaruSpec]:
    """Tokenizer and audio spec of the served checkpoint (blocking file / Hub reads)."""
    return load_tokenizer(model_path), HaruSpec.from_file(checkpoint_file(model_path, "config.json"))


class HaruDuplexPlugin(DuplexModelPlugin):
    plugin_id = "haru"
    private_runtime_config_keys = frozenset({"haru", "haru_prompt_slots", "haru_sampling"})
    silence_continuation_samples = FRAME_SAMPLES
    silence_continuation_sample_rate_hz = SAMPLE_RATE

    def __init__(self, encode_audio) -> None:
        super().__init__(encode_audio)
        self.data_plane = HaruDataPlane(encode_audio)
        self._tokenizer = None
        self._guidance_range = (1.0, 2.0)
        self._load_lock = asyncio.Lock()

    # ------------------------------------------------------ engine policy

    def configure_sampling_params(
        self, *, runtime_config: dict[str, object], defaults: tuple[object, ...]
    ) -> tuple[object, ...]:
        if not defaults or not isinstance(defaults[0], SamplingParams):
            return defaults
        params = defaults[0].clone()
        params.max_tokens = 1
        params.output_kind = RequestOutputKind.DELTA
        params.repetition_penalty = 1.0
        params.detokenize = False
        sampling = runtime_config.get("haru_sampling")
        for name, value in sampling.items() if isinstance(sampling, Mapping) else ():
            setattr(params, name, value)
        return (params, *defaults[1:])

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
        decode_pcm_f32le_payload(payload, sample_rate_hz=SAMPLE_RATE, exact_samples=FRAME_SAMPLES, model="Haru")
        first = seq <= 1
        duplex: dict[str, object] = {
            "data_plane": True,
            "session_id": fence.session_id,
            "epoch": fence.epoch,
            "seq": seq,
            "payload": dict(payload),  # type: ignore[call-overload]
        }
        if first:
            duplex["runtime_config"] = {"haru": runtime_config.get("haru")}
        slots = int(runtime_config.get("haru_prompt_slots", 1)) if first else 1  # type: ignore[call-overload]
        return DuplexAppendPlan(
            prompt={
                "prompt_token_ids": [0] * slots,
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "global_request_id": [fence.session_id],
                    "duplex": duplex,
                },
            }
        )

    def decide_output(self, **_: object) -> DuplexOutputDecision | None:
        # The model itself decides when to speak; every frame's audio is sent.
        return None

    # ----------------------------------------------------- session policy

    def create_session_state(self) -> HaruSessionState:
        return HaruSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return _capabilities(max_sessions)

    async def _load(self, model_config: ModelConfig | None) -> None:
        if self._tokenizer is not None:
            return
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            raise DuplexRuntimeConfigError("the model path is unavailable", code="model_path_unavailable")
        async with self._load_lock:
            if self._tokenizer is None:
                # Both reads may hit the Hub for a repo id: keep them off the event loop.
                tokenizer, config = await asyncio.to_thread(_load_checkpoint, model_path)
                # Prompt slots and the PCM framing use these constants; the worker uses the checkpoint's.
                if (config.sample_rate, config.frame_samples) != (SAMPLE_RATE, FRAME_SAMPLES):
                    raise DuplexRuntimeConfigError(
                        f"the checkpoint uses {config.frame_samples}-sample frames at {config.sample_rate} Hz; "
                        f"Haru duplex serving supports {FRAME_SAMPLES} at {SAMPLE_RATE} Hz",
                        code="unsupported_checkpoint",
                    )
                self._guidance_range = config.head.guidance_range
                self.data_plane.tokenizer = tokenizer
                self.data_plane.silent_ids = {config.pad_token_id, config.eop_token_id}
                self._tokenizer = tokenizer

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        await self._load(model_config)
        extra = _client_options(config.extra_body)
        low, high = self._guidance_range
        guidance = extra.get("guidance_scale", SessionOptions.guidance_scale)
        if not low <= guidance <= high:
            raise DuplexRuntimeConfigError(f"extra_body.guidance_scale must be within [{low}, {high}]")
        _check_ranges(extra, config.temperature)
        reference, reference_samples = await asyncio.to_thread(_decode_reference, config.ref_audio)
        options = SessionOptions(
            system_prompt=config.instructions or "",
            reference_pcm16=reference,
            guidance_scale=guidance,
            eop_bias=extra.get("eop_bias", 0.0),
            speak_first=extra.get("speak_first", False),
            dry=extra.get("dry", True),
            seed=extra.get("seed"),
        )
        sampling = {k: extra[k] for k in ("top_k", "top_p") if k in extra}
        if config.temperature is not None:
            sampling["temperature"] = float(config.temperature)
        slots = prompt_length(
            self._tokenizer, options.system_prompt, reference_samples, options.speak_first, FRAME_SAMPLES
        )
        return {"haru": asdict(options), "haru_prompt_slots": slots, "haru_sampling": sampling}

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        options = current.get("haru")
        if isinstance(options, Mapping) and (config.instructions or "") != options.get("system_prompt", ""):
            raise DuplexRuntimeConfigError(
                "instructions cannot be changed after the session is created", code="instructions_update_unsupported"
            )
        # Options and sampling are fixed when the session's request is created: refuse a change instead
        # of accepting an update that would not apply. (The framework already refuses a new ref_audio.)
        sampling = current.get("haru_sampling")
        sampling = sampling if isinstance(sampling, Mapping) else {}
        created = dict(options) if isinstance(options, Mapping) else {}
        created.update({k: sampling[k] for k in ("top_k", "top_p") if k in sampling})
        requested = _client_options(config.extra_body)
        changed = [name for name, value in requested.items() if created.get(name) != value]
        if (None if config.temperature is None else float(config.temperature)) != sampling.get("temperature"):
            changed.append("temperature")
        if changed:
            raise DuplexRuntimeConfigError(
                f"{', '.join(changed)} cannot be changed after the session is created", code="option_update_unsupported"
            )
        return deepcopy(dict(current))
