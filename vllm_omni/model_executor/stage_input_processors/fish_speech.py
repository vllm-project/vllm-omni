"""Stage input processor for Fish Speech S2 Pro: Slow AR → DAC Decoder."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)


def _get_connector_extra(transfer_manager: Any) -> dict[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    return raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}


def _use_tensor_code_payload(transfer_manager: Any) -> bool:
    cfg = _get_connector_extra(transfer_manager)
    return bool(cfg.get("fish_speech_tensor_codes", False))


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))


def _cfg_int(cfg: dict[str, Any], key: str, default: int = 0) -> int:
    value = cfg.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Fish Speech integer config {key}={value!r}") from exc


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Fish Speech float config {key}={value!r}") from exc


def _active_codec_requests(transfer_manager: Any) -> int:
    return sum(1 for value in transfer_manager.code_prompt_token_ids.values() if len(value) > 0)


def _ready_codec_requests(transfer_manager: Any, min_frames: int) -> int:
    return sum(1 for value in transfer_manager.code_prompt_token_ids.values() if len(value) >= min_frames)


def _select_backlog_chunk_size(
    transfer_manager: Any,
    cfg: dict[str, Any],
    base_chunk_size: int,
    current_length: int,
) -> int:
    backlog_chunk_size = _cfg_int(
        cfg,
        "fish_speech_backlog_codec_chunk_frames",
        0,
    )
    if backlog_chunk_size <= base_chunk_size:
        return base_chunk_size

    capacity = max(1, int(getattr(transfer_manager, "scheduler_max_num_seqs", 1)))
    active = _active_codec_requests(transfer_manager)
    load_factor = min(active / capacity, 1.0)
    threshold = _cfg_float(
        cfg,
        "fish_speech_backlog_load_threshold",
        0.75,
    )
    if load_factor < threshold:
        return base_chunk_size

    ready_min = _cfg_int(
        cfg,
        "fish_speech_backlog_min_ready",
        max(2, min(capacity, int(capacity * threshold))),
    )
    # Include the current request even when the caller has not yet committed
    # the latest frame into the transfer-manager map.
    ready = _ready_codec_requests(transfer_manager, base_chunk_size)
    if current_length >= base_chunk_size:
        ready = max(ready, 1)
    if ready < ready_min:
        return base_chunk_size

    return backlog_chunk_size


def _extract_last_frame(multimodal_output: dict[str, Any]) -> torch.Tensor | None:
    """Extract the last frame of audio codes from the multimodal output."""
    audio_codes = multimodal_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        valid = multimodal_output.get("audio_code_valid")
        if isinstance(valid, torch.Tensor) and valid.numel() > 0:
            is_valid = bool(valid.reshape(-1)[-1].item())
        elif valid is not None:
            is_valid = bool(valid)
        else:
            is_valid = bool(frame.any().item())
        if frame.numel() == 0 or not is_valid:
            return None
        return frame.to(device="cpu", dtype=torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(device="cpu", dtype=torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Fish Speech async_chunk: {tuple(audio_codes.shape)}")


def slow_ar_to_dac_decoder(
    source_outputs: list[Any],
    _prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async processor: wait for Slow AR to finish, then pass all codes to DAC decoder."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    slow_ar_outputs = source_outputs
    dac_inputs: list[OmniTokensPrompt] = []

    for output in slow_ar_outputs:
        out = output.outputs[0]
        # audio_codes shape: [num_frames, num_codebooks]
        audio_codes = out.multimodal_output["audio_codes"].to(torch.long)
        # Filter zero-padded frames.
        valid_mask = audio_codes.any(dim=1)
        audio_codes = audio_codes[valid_mask]
        # Codebook-major flat: [num_codebooks * num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        dac_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return dac_inputs


def slow_ar_to_dac_decoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async streaming processor: emit code chunks as they are produced.

    Accumulates per-step codes and emits fixed-size chunks with left context
    overlap for smooth audio transitions, analogous to
    ``talker2code2wav_async_chunk`` in Qwen3 TTS.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    cfg = _get_connector_extra(transfer_manager)

    if isinstance(multimodal_output, Mapping):
        frame = _extract_last_frame(multimodal_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.detach())
    elif not finished:
        return None

    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames", 0))

    initial_chunk_size = configured_initial_chunk_size

    # Per-request override.
    additional_information = getattr(request, "additional_information", None)
    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])

    if chunk_size <= 0 or left_context_size_config < 0 or configured_initial_chunk_size < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )
    if initial_chunk_size > chunk_size:
        initial_chunk_size = chunk_size

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    steady_chunk_size = _select_backlog_chunk_size(transfer_manager, cfg, chunk_size, length)

    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    single_initial_chunk = _cfg_bool(
        cfg,
        "fish_speech_single_initial_chunk",
        False,
    )
    use_first_chunk = initial_chunk_size > 0 and initial_chunk_size < steady_chunk_size

    if single_initial_chunk and use_first_chunk:
        if length <= initial_chunk_size:
            if not finished and length < initial_chunk_size:
                return None
            context_length = length if finished and length < initial_chunk_size else initial_chunk_size
        else:
            adjusted = length - initial_chunk_size
            if adjusted <= 0:
                return None
            if not finished and adjusted % steady_chunk_size != 0:
                return None
            chunk_length = adjusted % steady_chunk_size
            context_length = chunk_length if chunk_length != 0 else steady_chunk_size
        end_index = min(length, left_context_size_config + context_length)
        left_context_size = max(0, int(end_index - context_length))
        window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]
    elif initial_chunk_size > 0 and length <= chunk_size:
        already_sent = transfer_manager.put_req_chunk[request_id] * initial_chunk_size
        pending = length - already_sent
        if pending <= 0:
            return None
        if pending < initial_chunk_size and not finished:
            return None
        context_length = min(pending, initial_chunk_size)
        left_context_size = max(0, length - context_length)
        window_frames = transfer_manager.code_prompt_token_ids[request_id][:length]
    else:
        initial_coverage = (chunk_size // initial_chunk_size) * initial_chunk_size if initial_chunk_size > 0 else 0
        adjusted = length - initial_coverage
        chunk_length = adjusted % steady_chunk_size
        if chunk_length != 0 and not finished:
            return None
        context_length = chunk_length if chunk_length != 0 else steady_chunk_size
        end_index = min(length, left_context_size_config + context_length)
        left_context_size = max(0, int(end_index - context_length))
        window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack into codebook-major codes. The tensor path avoids expanding codec
    # indices into Python ints across the connector boundary; Stage1 schedules
    # a single placeholder token and consumes the real codes from runtime info.
    stacked_frames = torch.stack(window_frames, dim=0)
    codes_qf = stacked_frames.transpose(0, 1).contiguous()
    if _use_tensor_code_payload(transfer_manager):
        code_predictor_codes = codes_qf
    else:
        code_predictor_codes = codes_qf.reshape(-1)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )
