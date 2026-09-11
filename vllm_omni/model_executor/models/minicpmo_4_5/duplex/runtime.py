# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.engine.duplex.runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)

_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600
# <image> + 64 resampler embeddings + </image> per frame (max_slice_nums=1),
# matching MiniCPMO45DuplexPolicy.VISION_TOKENS_PER_FRAME.
_DUPLEX_VISION_TOKENS_PER_FRAME = 66
# Official stacked pair uses max_slice_nums=[2, 1]: the current frame is HD
# sliced (1 source + 2 patches on 960x540) and the composite is not.
_DUPLEX_HD_SLICES_PER_BASE_FRAME = 3


def _duplex_frame_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    frames = payload.get("video_frames")
    if not isinstance(frames, list):
        return 0
    return sum(1 for frame in frames if isinstance(frame, str) and frame)


def _duplex_vision_tokens(payload: object) -> int:
    """Scheduler slots for this append's camera track.

    Audio is never stacked: a unit still carries one second of soundtrack.
    ``stack_frames`` only adds a second *image*. Official HD on that pair is
    ``[2, 1]``, so the base frame reserves three 66-token blocks and every
    extra frame reserves one.
    """
    count = _duplex_frame_count(payload)
    if count <= 0:
        return 0
    if count >= 2:
        return (_DUPLEX_HD_SLICES_PER_BASE_FRAME + (count - 1)) * _DUPLEX_VISION_TOKENS_PER_FRAME
    return count * _DUPLEX_VISION_TOKENS_PER_FRAME


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    sample_count = _duplex_pcm_sample_count(payload)
    return sample_count is not None and sample_count != 0 and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_first_append_unit_count(payload: object) -> int | None:
    sample_count = _duplex_pcm_sample_count(payload)
    if not sample_count or sample_count % _DUPLEX_CHUNK_SAMPLES != 0:
        return None
    return max(1, sample_count // _DUPLEX_CHUNK_SAMPLES - 1)


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    vision_tokens = _duplex_vision_tokens(payload)
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default)) + vision_tokens
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN) + vision_tokens
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)) + vision_tokens


def duplex_first_append_context_reserve(runtime_config: object) -> int:
    if not isinstance(runtime_config, dict):
        return 48
    exact = runtime_config.get("duplex_first_append_context_tokens")
    if isinstance(exact, int) and exact >= 0:
        return exact
    reserve = 48
    ref = runtime_config.get("ref_audio_data")
    if isinstance(ref, str) and ref:
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            raw = b""
        if raw:
            reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
    return reserve


def _duplex_force_listen_count(extra_body: object) -> int:
    raw = extra_body.get("force_listen_count") if isinstance(extra_body, dict) else None
    try:
        return 0 if raw is None else max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def build_duplex_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    runtime_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
) -> dict[str, Any]:
    control = isinstance(payload, dict) and payload.get("gander_control") is True
    if control:
        ids = payload.get("token_ids")
        replay = payload.get("gander_replay") is True
        wake = payload.get("gander_wake") is True
        if not isinstance(ids, list) or len(ids) > 1500 or (not replay and ((not ids and not wake) or seq <= 1)):
            raise ValueError("Gander context append requires initialized session and bounded token ids")
        token_budget = len(ids) + (1 if seq == 1 else 3)
        if seq == 1:
            token_budget += duplex_first_append_context_reserve(runtime_config)
    else:
        token_budget = duplex_scheduler_token_budget(payload)
    if not control and seq <= 1:
        context_reserve = duplex_first_append_context_reserve(runtime_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = context_reserve + first_units * 12 - 1 + _duplex_vision_tokens(payload)
    if seq > 1 and duplex_payload_is_exact_chunks(payload):
        token_budget += 1
    if isinstance(payload, dict) and payload.get("gander_replay"):
        token_budget += max(0, len(payload.get("gander_replay_output_ids", [])) - 1)
    # final arms the model's turn-end fence; it does not build another audio
    # unit. Reserving an extra 12 slots here used to execute phantom padding
    # before every committed tail (25 scheduled tokens for 13 real embeddings).
    extra_body = session_config.get("extra_body")
    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        token_id = 0 if raw_token_id is None else max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}
    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "incarnation": fence.incarnation,
                "epoch": fence.epoch,
                "seq": seq,
                "gander_unit_id": (payload.get("gander_unit_id") if isinstance(payload, dict) else None)
                or f"u{fence.epoch}-{seq}",
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "data_plane": True,
                "recovery_replay": False,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": token_budget,
                "scheduler_token_id": token_id,
            },
        },
    }


def _coerce_int(value: object) -> int | None:
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            flat: Any = detach().cpu().reshape(-1)
            if flat.numel() == 0:
                return None
            value = flat[0].item()
        except Exception:
            return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [token_id for item in value if (token_id := _coerce_int(item)) is not None]


def _first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    return outputs[0] if isinstance(outputs, list) and outputs else None


def _multimodal_output(output: object, completion: object | None) -> dict[str, Any]:
    metadata = getattr(output, "multimodal_output", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    metadata = getattr(completion, "multimodal_output", None) if completion is not None else None
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _special_token_ids(metadata: dict[str, Any]) -> dict[str, int]:
    sources: list[object] = [metadata.get("special_token_ids"), metadata.get("meta")]
    sources.append(
        {
            key.removeprefix("meta."): value
            for key, value in metadata.items()
            if isinstance(key, str) and key.startswith("meta.")
        }
    )
    token_ids: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(key, str) and key.startswith("gander_"):
                from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import latest_int

                token_id = latest_int(value)
            else:
                token_id = _coerce_int(value)
            if isinstance(key, str) and token_id is not None and token_id >= 0:
                token_ids[key] = token_id
    return token_ids


def _completion_token_ids(completion: object | None) -> list[int]:
    if completion is None:
        return []
    for attribute in ("token_ids", "cumulative_token_ids"):
        token_ids = _coerce_int_list(getattr(completion, attribute, None))
        if token_ids:
            return token_ids
    return []


def _stage_config_value(runtime_config: dict[str, Any], key: str, stage_id: int) -> object | None:
    raw = runtime_config.get(key)
    if isinstance(raw, dict):
        value = raw.get(stage_id)
        return raw.get(str(stage_id)) if value is None else value
    if isinstance(raw, (list, tuple)) and stage_id < len(raw):
        return raw[stage_id]
    return None


class MiniCPMO45DuplexRuntimeExtension:
    adapter_id = "minicpmo45"
    runtime_extension_id = "minicpmo45"

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        path = runtime_config.get("gander_tokenizer_path")
        if isinstance(path, str):
            from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import tokenizer_for

            self._gander_tokenizer = tokenizer_for(path)
        configured: list[object] = []
        for stage_id, default in enumerate(defaults):
            max_tokens = _coerce_int(_stage_config_value(runtime_config, "duplex_stage_max_tokens", stage_id))
            raw_overrides = _stage_config_value(runtime_config, "duplex_stage_sampling_params", stage_id)
            overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
            if not isinstance(default, SamplingParams) or (not overrides and (max_tokens is None or max_tokens <= 0)):
                configured.append(default)
                continue
            params = default.clone()
            if max_tokens is not None and max_tokens > 0:
                params.max_tokens = max_tokens
            for name, value in overrides.items():
                if not hasattr(params, name):
                    continue
                setattr(params, name, value)
                if name == "stop_token_ids":
                    all_stop_token_ids = getattr(params, "_all_stop_token_ids", None)
                    if isinstance(all_stop_token_ids, set):
                        all_stop_token_ids.update(int(token_id) for token_id in value)
            configured.append(params)
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        return DuplexAppendPlan(
            prompt=build_duplex_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
            )
        )

    @staticmethod
    def automatic_rollover_allowed(payload, runtime_config):
        return not (
            runtime_config.get("gander_enabled")
            and isinstance(payload, dict)
            and payload.get("gander_defer_rollover") is True
        )

    @staticmethod
    def context_window_due(prompts, runtime_config):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import should_rollover

        return bool(runtime_config.get("gander_enabled")) and should_rollover(prompts, runtime_config)

    @staticmethod
    def select_context_window(prompts, runtime_config):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import select_units, unit_id

        if not runtime_config.get("gander_enabled"):
            return None
        retained = {unit_id(p) for p in select_units(prompts, runtime_config, compact=True)}
        return tuple(i for i, p in enumerate(prompts) if unit_id(p) in retained)

    @staticmethod
    def can_evict_context_unit(prompt):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import metadata

        return not metadata(prompt).get("gander_pinned", False)

    def prepare_context_replay(self, *, prompt, request_id, initial, runtime_config, fence, recovery_replay=True):
        if not runtime_config or not runtime_config.get("gander_enabled"):
            return self.prepare_recovery_prompt(prompt=prompt, request_id=request_id, initial=initial)
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import metadata

        old = metadata(prompt)
        payload = deepcopy(old["payload"])
        payload.update(
            gander_replay=recovery_replay,
            force_listen=True if recovery_replay else bool(payload.get("force_listen", False)),
            gander_replay_output_ids=list(old.get("gander_output_ids", [])) if recovery_replay else [],
            context_version=int(runtime_config.get("gander_context_version", 0)),
        )
        seq = int(old["seq"])
        rebuilt = build_duplex_data_plane_prompt(
            request_id=request_id,
            fence=fence or old["fence"],
            session_config=old["session_config"],
            runtime_config=runtime_config,
            seq=1 if initial else max(2, seq),
            turn_seq=int(old.get("turn_seq", seq)),
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
        )
        metadata(rebuilt).update(
            seq=seq, gander_unit_id=old.get("gander_unit_id"), gander_output_ids=list(old.get("gander_output_ids", []))
        )
        return rebuilt

    def plan_context_replacement(self, **kwargs):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import make_plan

        if not kwargs["runtime_config"].get("gander_enabled"):
            raise ValueError("context replacement is only enabled for Gander")
        return make_plan(**kwargs)

    @staticmethod
    def describe_context(prompts):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import describe

        return [describe(p) for p in prompts]

    def finalize_context_unit(self, *, prompts, output, segment_token_ids, segment_output_metadata):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import current_unit, latest_int

        completion = _first_completion(output)
        output_meta = _multimodal_output(output, completion)
        if any(
            latest_int(source.get(key))
            for source in (output_meta, segment_output_metadata)
            for key in ("duplex_recovery_replay", "meta.duplex_recovery_replay")
        ):
            return None
        ids = _special_token_ids(segment_output_metadata)
        ids.update(_special_token_ids(output_meta))
        seq = ids.get("gander_append_seq")
        if seq is None:
            return None
        candidates = [
            list(segment_token_ids),
            _completion_token_ids(completion),
            _coerce_int_list(getattr(completion, "cumulative_token_ids", None)),
        ]
        all_ids = max(candidates, key=len)
        terminal = latest_int(getattr(completion, "stop_reason", None))
        if terminal is None and all_ids:
            terminal = all_ids[-1]
        if terminal is None:
            return None
        unit = current_unit(all_ids, ids, finished=True)
        if not unit or unit[-1] != terminal:
            unit.append(terminal)
        from vllm_omni.engine.duplex.contracts import DuplexContextOutput

        return DuplexContextOutput(unit_sequence=seq, data={"output_ids": unit})

    @staticmethod
    def context_unit_sequence(prompt):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import metadata

        return metadata(prompt).get("seq")

    @staticmethod
    def apply_context_output(prompt, output):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import metadata

        updated = deepcopy(dict(prompt))
        metadata(updated)["gander_output_ids"] = list(output.data["output_ids"])
        return updated

    def prepare_recovery_prompt(
        self,
        *,
        prompt: dict[str, Any],
        request_id: str,
        initial: bool,
    ) -> dict[str, Any]:
        """Rebase a journal unit onto a new physical scheduler request.

        A rollover may discard the original first append.  The first retained
        unit must therefore reserve the model's session-prefix embeddings even
        though its logical ``seq`` remains unchanged for fencing/idempotency.
        """
        copied = deepcopy(prompt)
        model_buffer = copied.get("model_intermediate_buffer")
        duplex = model_buffer.get("duplex") if isinstance(model_buffer, dict) else None
        if not isinstance(duplex, dict):
            raise ValueError("MiniCPM-o recovery journal entry has no duplex metadata")
        fence = duplex.get("fence")
        if not isinstance(fence, DuplexFence):
            raise ValueError("MiniCPM-o recovery journal entry has no typed fence")
        try:
            mode = DuplexInputMode(duplex["mode"])
            seq = int(duplex["seq"])
            turn_seq = int(duplex["turn_seq"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("MiniCPM-o recovery journal entry has invalid sequence metadata") from exc
        session_config = duplex.get("session_config")
        runtime_config = duplex.get("runtime_config")
        payload = duplex.get("payload")
        if not isinstance(session_config, dict) or not isinstance(runtime_config, dict):
            raise ValueError("MiniCPM-o recovery journal entry has invalid runtime configuration")
        rebuilt = build_duplex_data_plane_prompt(
            request_id=request_id,
            fence=fence,
            session_config=session_config,
            runtime_config=runtime_config,
            seq=(1 if initial else seq),
            turn_seq=turn_seq,
            mode=mode,
            payload=payload,
            final=bool(duplex.get("final")),
        )
        rebuilt_duplex = rebuilt["model_intermediate_buffer"]["duplex"]
        # Preserve logical ordering even when the physical first unit is
        # materialized with first-append token budgeting.
        rebuilt_duplex["seq"] = seq
        return rebuilt

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        if stage_id >= final_stage_id or not segment_finished:
            return None

        completion = _first_completion(output)
        output_metadata = _multimodal_output(output, completion)
        special_token_ids = _special_token_ids(segment_output_metadata)
        special_token_ids.update(_special_token_ids(output_metadata))
        if "tool_call_token_id" in special_token_ids:
            from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import current_unit

            # A DELTA completion / segment may contain only the final token.
            # Tool parsing needs the whole current unit, including its opener.
            candidates = [list(segment_token_ids), _completion_token_ids(completion)]
            candidates.append(_coerce_int_list(getattr(completion, "cumulative_token_ids", None)))
            tokens = current_unit(max(candidates, key=len), special_token_ids, finished=True)
            if tokens and tokens[0] == special_token_ids["tool_call_token_id"]:
                tokenizer = getattr(self, "_gander_tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError("Gander tool output tokenizer is unavailable")
                raw = tokenizer.decode(tokens, skip_special_tokens=False)
                import hashlib

                identity = f"{getattr(output, 'request_id', '')}:{special_token_ids.get('gander_append_seq')}:{raw}"
                return DuplexOutputDecision(
                    action=DuplexOutputAction.DIRECT_RESPONSE,
                    # Serving ends the turn for silent tool units as well.
                    # Advance before already-queued input can produce speech.
                    ends_model_turn=True,
                    metadata={
                        **output_metadata,
                        **{f"meta.{k}": v for k, v in special_token_ids.items()},
                        "duplex_direct_response": True,
                        "gander_tool_text": raw,
                        "gander_call_id": "call_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    },
                )

        listen_id = special_token_ids.get("listen_token_id")
        if listen_id is None:
            return None

        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        token_ids = _completion_token_ids(completion) or list(segment_token_ids)
        final_token = _coerce_int(stop_reason)
        if final_token is None and token_ids:
            final_token = token_ids[-1]
        interrupt_id = special_token_ids.get("interrupt_token_id")
        interrupted = interrupt_id is not None and final_token == interrupt_id
        if final_token != listen_id and not interrupted:
            return None

        metadata = dict(output_metadata)
        for key, value in special_token_ids.items():
            metadata.setdefault(f"meta.{key}", value)
        metadata.update(
            {
                "duplex_direct_response": True,
                "duplex_native_decision": "interrupt" if interrupted else "listen",
                "model_listen": True,
                "listen_source": "model_listen",
            }
        )
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
            ends_model_turn=interrupted,
        )


__all__ = [
    "MiniCPMO45DuplexRuntimeExtension",
    "build_duplex_data_plane_prompt",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_payload_is_exact_chunks",
    "duplex_scheduler_token_budget",
]
