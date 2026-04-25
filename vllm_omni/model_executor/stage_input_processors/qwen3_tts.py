"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = init_logger(__name__)


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            # Non-async decode should only run once, after talker has
            # accumulated the final code sequence.
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})
        # audio_codes shape: [num_frames, Q] where Q=num_quantizers (16)
        audio_codes = mm_codes["audio"].to(torch.long)
        token_ids = output.token_ids
        # token_ids provides an upper bound on the newly generated codec span.
        # audio_codes may still contain zero-padded / invalid rows, so trim only
        # after filtering valid frames instead of trying to align EOS indices.
        seq_len = max(len(token_ids) - 1, 0)
        # Filter invalid frames: zero-padded (EOS) and frames containing
        # out-of-range values (e.g. stop_token_id=2150 exceeds codebook_size=2048).
        _CODEBOOK_SIZE = 2048
        valid_mask = audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
        audio_codes = audio_codes[valid_mask]
        if seq_len > 0 and audio_codes.ndim == 2 and int(audio_codes.shape[0]) > seq_len:
            audio_codes = audio_codes[-seq_len:]
        ref_code = mm_codes.get("ref")
        ref_code_len = mm.get("meta", {}).get("ref_code_len")
        if isinstance(ref_code_len, torch.Tensor):
            ref_code_len = int(ref_code_len.reshape(-1)[-1].item()) if ref_code_len.numel() > 0 else 0
        elif ref_code_len is None:
            ref_code_len = 0
        else:
            ref_code_len = int(ref_code_len)
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            ref_code = ref_code.to(torch.long).cpu().contiguous()
            if ref_code.ndim == 1:
                num_quantizers = int(audio_codes.shape[1]) if audio_codes.ndim == 2 and audio_codes.shape[1] > 0 else 16
                if ref_code.numel() % num_quantizers != 0:
                    logger.warning(
                        "Ignoring malformed ref_code with %d elements not divisible by num_quantizers=%d",
                        ref_code.numel(),
                        num_quantizers,
                    )
                    ref_code = None
                else:
                    ref_code = ref_code.reshape(-1, num_quantizers)
            elif ref_code.ndim != 2:
                logger.warning("Ignoring malformed ref_code shape %s", tuple(ref_code.shape))
                ref_code = None
            if isinstance(ref_code, torch.Tensor) and ref_code_len > 0 and int(ref_code.shape[0]) > ref_code_len:
                logger.warning(
                    "Trimming ref_code from %d frames to ref_code_len=%d before Code2Wav.",
                    int(ref_code.shape[0]),
                    ref_code_len,
                )
                ref_code = ref_code[:ref_code_len]
            if not isinstance(ref_code, torch.Tensor):
                ref_code_len = 0
            else:
                ref_code_len = int(ref_code.shape[0])
                audio_codes = torch.cat([ref_code.to(audio_codes.device), audio_codes], dim=0)
        else:
            ref_code_len = 0
        # Code2Wav expects codebook-major flat: [Q*num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        additional_information: dict[str, Any] = {}
        if ref_code_len > 0:
            additional_information["meta"] = {"left_context_size": [ref_code_len]}
        # Propagate speaker and language from the original prompt so they are
        # available as runtime_additional_information in later pipeline stages,
        # consistent with qwen3-omni and qwen2.5-omni stage input processors.
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            additional_information["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            additional_information["language"] = language
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information if additional_information else None,
            )
        )
    return code2wav_inputs


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    audio_codes = pooling_output.get("codes", {}).get("audio")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Qwen3-TTS async_chunk: {tuple(audio_codes.shape)}")


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            codec_codes = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
        ref_code = pooling_output.get("codes", {}).get("ref")
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0 and request_payload.get(request_id) is None:
            request_payload[request_id] = ref_code.to(torch.long).cpu().contiguous()
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    # Per-request override takes priority over dynamic IC.
    per_request_override = False
    initial_chunk_size = 0
    additional_information = getattr(request, "additional_information", None)

    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
            per_request_override = True

    # Dynamic IC: cache per request so boundaries stay stable for its lifetime.
    if not per_request_override:
        _ic_cache = getattr(transfer_manager, "_cached_ic", None)
        if _ic_cache is None:
            _ic_cache = {}
            transfer_manager._cached_ic = _ic_cache
        if request_id not in _ic_cache:
            max_ic = max_ic_for_chunk_size(chunk_size)
            active = sum(1 for v in transfer_manager.code_prompt_token_ids.values() if len(v) > 0)
            capacity = getattr(transfer_manager, "scheduler_max_num_seqs", 1)
            _ic_cache[request_id] = compute_dynamic_initial_chunk_size(active, capacity, max_ic)
        initial_chunk_size = _ic_cache[request_id]

    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )

    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping to codec_chunk_frames.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {
                "codes": {"audio": []},
                "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
            }
        return None

    in_initial_phase = initial_chunk_size > 0 and initial_chunk_size < chunk_size and length <= chunk_size

    if in_initial_phase:
        # IC phase: emit every initial_chunk_size frames with growing left context.
        if not finished and length % initial_chunk_size != 0:
            return None
        context_length = (
            length % initial_chunk_size if (finished and length % initial_chunk_size != 0) else initial_chunk_size
        )
    else:
        # Normal phase: offset so the first normal emit picks up after IC phase.
        # IC is stateless (may change with load); any mismatch is absorbed by left_context.
        initial_coverage = (
            (chunk_size // initial_chunk_size) * initial_chunk_size if 0 < initial_chunk_size < chunk_size else 0
        )
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Prepend ref_code as decoder context for every chunk so the vocoder
    # maintains voice-clone speaker identity throughout the stream.  The HF
    # reference decodes ref_code + all_codes in one pass; without ref_code
    # context on later chunks the decoder loses speaker identity and produces
    # distorted audio.  Use `.get()` (not `.pop()`) to keep ref_code for
    # subsequent chunks.
    ref_code = request_payload.get(request_id)
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        ref_frames = ref_code.tolist()
        window_frames = ref_frames + window_frames
        left_context_size += len(ref_frames)

    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)
    code_predictor_codes = [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)]

    info: dict[str, Any] = {
        "codes": {"audio": code_predictor_codes},
        "meta": {"left_context_size": left_context_size, "finished": torch.tensor(finished, dtype=torch.bool)},
    }
    # Propagate speaker and language from the request so they are available
    # as runtime_additional_information in subsequent pipeline stages, consistent
    # with qwen3-omni and qwen2.5-omni stage input processors.
    speaker = extract_speaker_from_request(request)
    if speaker is not None:
        info["speaker"] = speaker
    language = extract_language_from_request(request)
    if language is not None:
        info["language"] = language
    return info
