# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from base64 import b64decode
from binascii import Error as BinasciiError
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
# sliced (1 source + up to 2 patches, e.g. on 960x540) and the composite is
# not. The patch count depends on the frame size (see
# ``_duplex_hd_slice_count``); the constant is the fallback when the frame
# header cannot be read.
_DUPLEX_HD_SLICES_PER_BASE_FRAME = 3
_DUPLEX_HD_MAX_SLICE_NUMS = 2
_DUPLEX_SCALE_RESOLUTION = 448


def _duplex_frames(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    frames = payload.get("video_frames")
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, str) and frame]


def _duplex_frame_count(payload: object) -> int:
    return len(_duplex_frames(payload))


def _duplex_frame_size(frame_b64: str) -> tuple[int, int] | None:
    """Pixel size of a base64 JPEG/PNG frame from its header, or ``None``."""
    from io import BytesIO

    try:
        from PIL import Image

        with Image.open(BytesIO(b64decode(frame_b64, validate=True))) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - Stage0 rejects bad frames with a reason
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _duplex_hd_slice_count(image_size: tuple[int, int], max_slice_nums: int) -> int:
    """Number of HD patches ``MiniCPMVImageProcessor.get_sliced_grid`` adds.

    Port of the official grid selection (``scale_resolution=448``): an image
    whose area fits in one 448x448 tile is not sliced at all, otherwise the
    grid closest to the image aspect ratio among ``multiple-1 .. multiple+1``
    splits is used. Stage0 runs the same processor, so the scheduler budget
    and the worker-built embeddings agree.
    """
    width, height = image_size
    ratio = width * height / (_DUPLEX_SCALE_RESOLUTION * _DUPLEX_SCALE_RESOLUTION)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return 0
    log_ratio = math.log(width / height)
    best_grid = (1, 1)
    min_error = float("inf")
    for split_grids_nums in (multiple - 1, multiple, multiple + 1):
        if split_grids_nums == 1 or split_grids_nums > max_slice_nums:
            continue
        for m in range(1, split_grids_nums + 1):
            if split_grids_nums % m:
                continue
            grid = (m, split_grids_nums // m)
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error
    return best_grid[0] * best_grid[1]


def _duplex_vision_tokens(payload: object) -> int:
    """Scheduler slots for this append's camera track.

    Audio is never stacked: a unit still carries one second of soundtrack.
    ``stack_frames`` only adds a second *image*. Official HD on that pair is
    ``max_slice_nums=[2, 1]``: the base frame keeps its source block plus the
    HD patches the processor cuts for its size (two for a 960x540 camera
    frame, none for anything that fits in one 448x448 tile) and every extra
    frame is one block. A single frame is never sliced. The count must match
    the embeddings Stage0 builds exactly: surplus slots become pad
    embeddings inside the KV.
    """
    frames = _duplex_frames(payload)
    count = len(frames)
    if count <= 0:
        return 0
    if count == 1:
        return _DUPLEX_VISION_TOKENS_PER_FRAME
    base_size = _duplex_frame_size(frames[0])
    if base_size is None:
        base_blocks = _DUPLEX_HD_SLICES_PER_BASE_FRAME
    else:
        base_blocks = 1 + _duplex_hd_slice_count(base_size, _DUPLEX_HD_MAX_SLICE_NUMS)
    return (base_blocks + (count - 1)) * _DUPLEX_VISION_TOKENS_PER_FRAME


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
    token_budget = duplex_scheduler_token_budget(payload)
    if seq <= 1:
        context_reserve = duplex_first_append_context_reserve(runtime_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = context_reserve + first_units * 12 - 1 + _duplex_vision_tokens(payload)
    if seq > 1 and duplex_payload_is_exact_chunks(payload):
        token_budget += 1
    if final and duplex_payload_is_exact_chunks(payload):
        token_budget += 12
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
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "data_plane": True,
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
    if isinstance(metadata, dict):
        return metadata
    metadata = getattr(completion, "multimodal_output", None) if completion is not None else None
    return metadata if isinstance(metadata, dict) else {}


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
    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
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
        listen_id = special_token_ids.get("listen_token_id")
        if listen_id is None:
            return None

        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        token_ids = _completion_token_ids(completion) or list(segment_token_ids)
        if _coerce_int(stop_reason) != listen_id and (not token_ids or token_ids[-1] != listen_id):
            return None

        metadata = dict(output_metadata)
        for key, value in special_token_ids.items():
            metadata.setdefault(f"meta.{key}", value)
        metadata.update(
            {
                "duplex_direct_response": True,
                "duplex_native_decision": "listen",
                "model_listen": True,
                "listen_source": "model_listen",
            }
        )
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
        )


__all__ = [
    "MiniCPMO45DuplexRuntimeExtension",
    "build_duplex_data_plane_prompt",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_payload_is_exact_chunks",
    "duplex_scheduler_token_budget",
]
