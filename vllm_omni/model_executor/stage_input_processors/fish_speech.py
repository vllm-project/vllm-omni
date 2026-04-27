"""Stage input processor for Fish Speech S2 Pro: Slow AR → DAC Decoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    """Extract the last frame of audio codes from the pooling output."""
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Fish Speech async_chunk: {tuple(audio_codes.shape)}")


def slow_ar_to_dac_decoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async processor: wait for Slow AR to finish, then pass all codes to DAC decoder."""
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    slow_ar_outputs = _validate_stage_inputs(stage_list, engine_input_source)
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
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async streaming processor: emit code chunks as they are produced.

    Accumulates per-step codes and emits fixed-size chunks with left context
    overlap for smooth audio transitions, analogous to
    ``talker2code2wav_async_chunk`` in Qwen3 TTS.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.detach().to(device="cpu", dtype=torch.long))
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames", 0))

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

    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )
    if initial_chunk_size > chunk_size:
        initial_chunk_size = chunk_size

    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {
                "codes": {"audio": []},
                "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
            }
        return None

    in_initial_phase = initial_chunk_size > 0 and length <= chunk_size

    if in_initial_phase:
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
        chunk_length = adjusted % chunk_size
        if chunk_length != 0 and not finished:
            return None
        context_length = chunk_length if chunk_length != 0 else chunk_size
        end_index = min(length, left_context_size_config + context_length)
        left_context_size = max(0, int(end_index - context_length))
        window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack into codebook-major flat codes.
    stacked_frames = torch.stack(window_frames, dim=0)
    code_predictor_codes = stacked_frames.transpose(0, 1).reshape(-1).tolist()

    return {
        "codes": {"audio": code_predictor_codes},
        "meta": {"left_context_size": left_context_size, "finished": torch.tensor(finished, dtype=torch.bool)},
    }
