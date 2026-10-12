# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lychee binding for the engine-resident Unified Duplex session runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import TYPE_CHECKING

from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    EncodeAudio,
    PartialStageForward,
    reject_changed_runtime_value,
)

from .audio_encoding import make_lychee_audio_encoder
from .capabilities import lychee_native_capabilities
from .codec import LycheeCodecStreams, output_payload
from .data_plane import LycheeDataPlaneContext, LycheeDataPlaneSession
from .history import LycheeSessionHistory
from .input import TICKS_PER_WINDOW
from .session import LycheeServingSessionState

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def build_lychee_duplex_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, object],
    runtime_config: dict[str, object],
    seq: int,
    turn_seq: int,
    payload: object,
    final: bool,
) -> dict[str, object]:
    """Build the scheduler envelope; the native audio input bridge consumes it in P3/P4."""

    ticks = TICKS_PER_WINDOW
    if isinstance(payload, Mapping):
        ledger = payload.get("lychee_audio_ledger")
        if isinstance(ledger, Mapping):
            start = ledger.get("consumable_tick_start")
            end = ledger.get("consumable_tick_end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                ticks = end - start
    scheduler_token_id = runtime_config.get("duplex_scheduler_token_id", 0)
    if not isinstance(scheduler_token_id, int) or scheduler_token_id < 0:
        scheduler_token_id = 0
    return {
        # One causal input row starts this 400 ms segment. The ten logical
        # Lychee ticks are generated from it; audio embeddings are selected
        # request-locally by LycheeModelState as those ticks execute.
        "prompt_token_ids": [scheduler_token_id],
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "turn_seq": turn_seq,
                "mode": "append_audio_chunk",
                "payload": payload,
                "final": final,
                "data_plane": True,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": ticks,
                "scheduler_token_id": scheduler_token_id,
            },
        },
    }


class LycheeDuplexPlugin(DuplexModelPlugin):
    plugin_id = "lychee_fd"
    private_runtime_config_keys = frozenset(
        {
            "duplex_scheduler_token_id",
            "lychee_audio_pad_token_id",
            "lychee_audio_patch_token_id",
            "lychee_audio_sample_rate_hz",
            "lychee_audio_window_ms",
            "lychee_system_token_ids",
            "lychee_history",
            "lychee_audio_delta",
            "lychee_kv_rebuild",
            "lychee_speech_pad_token_id",
            "lychee_sleep_token_id",
        }
    )
    silence_continuation_samples = 6400

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self._tokenizer = None
        self.histories: dict[str, LycheeSessionHistory] = {}
        self.codec_streams = LycheeCodecStreams()
        self.data_plane = LycheeDataPlaneSession(
            make_lychee_audio_encoder(encode_audio),
            codec_streams=self.codec_streams,
            histories=self.histories,
            decode_text=self._decode_text,
        )

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        configured: list[object] = []
        for default in defaults:
            clone = getattr(default, "clone", None)
            params = clone() if callable(clone) else deepcopy(default)
            if hasattr(params, "output_kind"):
                params.output_kind = RequestOutputKind.DELTA
            if len(defaults) > 1 and len(configured) > 0:
                configured.append(params)
                continue
            if hasattr(params, "max_tokens"):
                params.max_tokens = TICKS_PER_WINDOW
            if hasattr(params, "min_tokens"):
                params.min_tokens = TICKS_PER_WINDOW
            if hasattr(params, "ignore_eos"):
                params.ignore_eos = True
            configured.append(params)
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
        if not isinstance(payload, Mapping):
            raise ValueError("Lychee append requires native PCM evidence")
        history = self.histories.get(fence.session_id)
        if history is None:
            prefix = runtime_config.get("lychee_system_token_ids")
            if not isinstance(prefix, list) or not prefix:
                raise ValueError("Lychee initial append requires a tokenized system prompt")
            history = self.histories[fence.session_id] = LycheeSessionHistory(
                prefix,
                text_pad=int(runtime_config.get("duplex_scheduler_token_id", 158358)),
                speech_pad=int(runtime_config.get("lychee_speech_pad_token_id", 158359)),
                sleep=int(runtime_config.get("lychee_sleep_token_id", 158357)),
            )
        resident = history.has_resident_binding(request_id=request_id, execution_epoch=fence.epoch)
        eos_rebuild = resident and history.pending_eos_rebuild_tick is not None
        rebinding = not resident or eos_rebuild
        prepared_payload = history.append_audio(payload, epoch=fence.epoch, seq=seq)
        generation_budget = (len(history.audio_windows) * 10 - history.frontier_tick - 1) if rebinding else 10
        if generation_budget <= 0:
            raise ValueError("Lychee append contains no new consumable logical ticks")
        prompt = build_lychee_duplex_prompt(
            request_id=request_id,
            fence=fence,
            session_config=session_config,
            runtime_config=runtime_config,
            seq=seq,
            turn_seq=turn_seq,
            payload=prepared_payload,
            final=final,
        )
        if rebinding:
            prompt["prompt_token_ids"] = list(history.text)
        bridge = prompt["model_intermediate_buffer"]["duplex"]
        if rebinding:
            # Initial/recovered KV needs all committed channel and PCM evidence.
            bridge["lychee_history"] = history.snapshot(execution_epoch=fence.epoch)
            if eos_rebuild:
                prompt["model_intermediate_buffer"]["meta"] = {"replace_streaming_prompt": True}
                bridge["lychee_kv_rebuild"] = {
                    "reason": "natural_speech_eos",
                    "eos_tick": history.pending_eos_rebuild_tick,
                    "frontier_tick": history.frontier_tick,
                }
        else:
            # Same-owner MRV2 state retains its cursor and previous/current
            # audio windows across buffer replacement. Send only the new PCM
            # in bridge.payload, with explicit coordinates for strict admission.
            window = history.audio_windows[-1]
            window_seq = int(window["seq"])
            bridge["lychee_audio_delta"] = {
                "version": 1,
                "kind": "resident_append",
                "request_id": request_id,
                "session_epoch": fence.epoch,
                "execution_epoch": fence.epoch,
                "op_seq": seq,
                "audio_window_seq": window_seq,
                "previous_audio_window_seq": window_seq - 1,
                "start_tick": int(window["start_tick"]),
                "window_ticks": TICKS_PER_WINDOW,
            }
        bridge["scheduler_token_budget"] = generation_budget
        clone = getattr(sampling_params, "clone", None)
        params = clone() if callable(clone) else deepcopy(sampling_params)
        params.max_tokens = generation_budget
        params.min_tokens = generation_budget
        params.ignore_eos = True
        return DuplexAppendPlan(prompt=prompt, sampling_params=params)

    def project_request_error(self, *, stage_id: int, request_id: str, error: str) -> dict[str, object] | None:
        if stage_id != 0 or "transaction aborted; rebuild required;" not in error:
            return None
        return {
            "data_plane_request_id": request_id,
            "error_code": "lychee_request_aborted",
            "error": error,
            "retryable": True,
            "recover_binding": True,
        }

    @staticmethod
    def _record_forced_listen_frontier(
        *, history: LycheeSessionHistory, request_id: str, fence: DuplexFence, plan: DuplexAppendPlan
    ) -> None:
        information = plan.prompt["model_intermediate_buffer"]
        bridge = information["duplex"]
        snapshot = bridge.get("lychee_history")
        if not isinstance(snapshot, dict) or snapshot.get("force_listen_at_frontier") is not True:
            return
        if (
            information.get("request_id") != request_id
            or bridge.get("fence") != fence
            or bridge.get("session_id") != fence.session_id
            or bridge.get("epoch") != fence.epoch
            or snapshot.get("execution_epoch") != fence.epoch
        ):
            raise ValueError("Lychee accepted forced-listen snapshot has a mismatched owner")
        ticks = snapshot.get("logical_ticks")
        if not isinstance(ticks, list) or not ticks or type(ticks[-1]) is not int or ticks[-1] < 0:
            raise ValueError("Lychee accepted forced-listen snapshot has an invalid frontier")
        position = len(ticks) - 1
        fields = ("text_input_ids", "speech_input_ids", "control_input_ids")
        channels = (history.text, history.speech, history.control)
        if (
            position >= len(history.ticks)
            or history.ticks[position] != ticks[-1]
            or any(not isinstance(snapshot.get(field), list) or len(snapshot[field]) != len(ticks) for field in fields)
            or any(position >= len(channel) for channel in channels)
        ):
            raise ValueError("Lychee accepted forced-listen snapshot no longer matches its canonical coordinate")
        expected = tuple(snapshot[field][position] for field in fields)
        actual = tuple(channel[position] for channel in channels)
        forced = (history.text_pad, history.speech_pad, history.sleep)
        if actual != expected and actual != forced:
            raise ValueError("Lychee accepted forced-listen snapshot has a replaced canonical frontier")
        # Submission may have yielded while later outputs advanced history.
        # Commit the exact input row the worker forced, retaining every raw
        # sample for historical merge conditioning and all later channel rows.
        for channel, value in zip(channels, forced):
            channel[position] = value

    def record_accepted_append_plan(self, *, request_id: str, fence: DuplexFence, plan: DuplexAppendPlan) -> None:
        """Retain accepted input facts even if cancellation requires compensation."""
        history = self.histories.get(fence.session_id)
        if history is not None:
            self._record_forced_listen_frontier(history=history, request_id=request_id, fence=fence, plan=plan)

    def commit_append_plan(self, *, request_id: str, fence: DuplexFence, plan: DuplexAppendPlan) -> None:
        history = self.histories[fence.session_id]
        self.record_accepted_append_plan(request_id=request_id, fence=fence, plan=plan)
        bridge = plan.prompt["model_intermediate_buffer"]["duplex"]
        rebuild = bridge.get("lychee_kv_rebuild")
        if isinstance(rebuild, dict) and rebuild.get("eos_tick") == history.pending_eos_rebuild_tick:
            history.pending_eos_rebuild_tick = None
        elif "lychee_history" in bridge and not history.has_resident_binding(
            request_id=request_id, execution_epoch=fence.epoch
        ):
            # Submission may yield while this fresh owner emits a later EOS.
            # Only EOS inputs covered by the accepted snapshot were rebuilt.
            frontier = bridge["lychee_history"]["logical_ticks"][-1]
            if history.pending_eos_rebuild_tick is not None and history.pending_eos_rebuild_tick <= frontier:
                history.pending_eos_rebuild_tick = None
        history.bind_request(request_id=request_id, execution_epoch=fence.epoch)
        history.force_listen_at_frontier = False

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

    def project_intermediate_output(self, *, stage_id: int, output: object, context: object) -> bool:
        return stage_id == 0

    def plan_partial_stage_output(self, orchestrator, stage_id, replica_id, output, req_state):
        if stage_id != 0 or req_state.final_stage_id <= stage_id:
            return None
        # Resident listening windows do not constitute waveform requests.
        # This also suppresses the orchestrator's segment-finished fallback.
        req_state.skip_legacy_stage_forward = True
        inner, completion, payload = output_payload(output)
        fence = req_state.stage_fences.get(stage_id)
        if fence is None or completion is None:
            return None
        packets = self.codec_streams.consume_all(output.request_id, payload, session_epoch=fence.epoch)
        plans = []
        for packet in packets:
            packet["codec_token_ids"] = packet.pop("codec_ids")
            forwarded = copy(inner)
            forwarded_completion = copy(completion)
            forwarded_completion.multimodal_output = {**payload, "lychee_t2w": packet}
            forwarded.outputs = [forwarded_completion]
            # EOF is model metadata; the native engine binding remains resident.
            plans.append(PartialStageForward(output=forwarded, is_final_update=False))
        if not plans:
            return None
        return PartialStageForward(output=plans[0].output, is_final_update=False, following_updates=tuple(plans[1:]))

    def discard_pending_input(self, *, session_id: str) -> None:
        history = self.histories.get(session_id)
        if history is not None:
            history.discard_pending_audio()

    def _decode_text(self, tokens: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Lychee text projection requires the model tokenizer")
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def create_session_state(self) -> LycheeServingSessionState:
        return LycheeServingSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return lychee_native_capabilities(max_sessions=max_sessions)

    def validate_client_extra_body(self, extra_body: object) -> None:
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ValueError("Lychee duplex extra_body must be an object")
        if isinstance(extra_body, dict):
            private_keys = sorted(self.private_runtime_config_keys.intersection(extra_body))
            if private_keys:
                raise ValueError("Lychee duplex runtime configuration is server-owned: " + ", ".join(private_keys))

    async def prepare_runtime_config(
        self,
        config: DuplexSessionConfig,
        *,
        model_config: ModelConfig | None,
    ) -> dict[str, object]:
        hf_config = getattr(model_config, "hf_config", None)
        prefix = None
        if model_config is not None:
            import asyncio

            from transformers import AutoTokenizer

            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, model_config.tokenizer, trust_remote_code=model_config.trust_remote_code
            )
            self._tokenizer = tokenizer
            instructions = config.instructions or "You are a helpful assistant."
            prefix = tokenizer.encode(f"<|BOT|>system\n{instructions}<|EOT|>", add_special_tokens=False)
        control_policy = {"allowing_backchannel": config.extra_body.get("allowing_backchannel", True)}
        if not isinstance(control_policy["allowing_backchannel"], bool):
            raise ValueError("allowing_backchannel must be a boolean")
        for name, default in (
            ("start_speak_token_factor", 1.2),
            ("start_listen_token_factor", 1.2),
            ("backchannel_token_bias", 1.0),
            ("end_speak_token_factor", 1.0),
        ):
            value = config.extra_body.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name != "backchannel_token_bias" and value <= 0:
                raise ValueError(f"{name} must be positive")
            control_policy[name] = float(value)
        return {
            **deepcopy(config.extra_body),
            **control_policy,
            "instructions": config.instructions,
            "lychee_system_token_ids": prefix,
            "lychee_speech_pad_token_id": int(getattr(hf_config, "stoken_pad_token_id", 158359)),
            "lychee_sleep_token_id": int(getattr(hf_config, "sleep_token_id", 158357)),
            "lychee_audio_sample_rate_hz": 16000,
            "lychee_audio_window_ms": 400,
            "lychee_audio_patch_token_id": int(getattr(hf_config, "audio_patch_token_id", 151_690)),
            "lychee_audio_pad_token_id": int(getattr(hf_config, "audio_pad_token_id", 158_360)),
            "duplex_scheduler_token_id": int(getattr(hf_config, "text_pad_token_id", 158_358)),
        }

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        reject_changed_runtime_value(
            config.instructions,
            current.get("instructions"),
            message="instructions cannot be changed after a Lychee session is created",
            code="instructions_update_unsupported",
        )
        return deepcopy(dict(current))

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> LycheeDataPlaneContext:
        return LycheeDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


__all__ = ["LycheeDuplexPlugin", "build_lychee_duplex_prompt"]
