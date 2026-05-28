"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)
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
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
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
        token_ids = output.cumulative_token_ids

        # token_ids provides an upper bound on the newly generated codec span.
        # audio_codes may still contain zero-padded / invalid rows, so trim only
        # after filtering valid frames instead of trying to align EOS indices.
        seq_len = max(len(token_ids) - 1, 0)
        # Filter invalid frames: zero-padded (EOS), out-of-range values (e.g.
        # stop_token_id=2150 exceeds codebook_size=2048), and negative
        # sentinels (e.g. -1 padding).
        _CODEBOOK_SIZE = 2048
        valid_mask = (
            (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
        )
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
        additional_information = to_dict(
            OmniPayloadStruct(
                meta=MetaStruct(left_context_size=ref_code_len) if ref_code_len > 0 else None,
                speaker=extract_speaker_from_prompt(prompt, index=i),
                language=extract_language_from_prompt(prompt, index=i),
            )
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information if additional_information else None,
            )
        )
    return code2wav_inputs


def _extract_last_frame(pooling_output: OmniPayload) -> torch.Tensor | None:
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
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
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
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)
    ref_code_context_frames = int(cfg.get("ref_code_context_frames") or left_context_size_config)

    # Per-request override takes priority over dynamic IC.
    fixed_initial_chunk_size = configured_initial_chunk_size > 0
    initial_chunk_size = configured_initial_chunk_size
    additional_information = getattr(request, "additional_information", None)

    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
            fixed_initial_chunk_size = True

    # Dynamic IC: cache per request so boundaries stay stable for its lifetime.
    if not fixed_initial_chunk_size:
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

    if (
        chunk_size <= 0
        or left_context_size_config < 0
        or configured_initial_chunk_size < 0
        or initial_chunk_size < 0
        or ref_code_context_frames < 0
    ):
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}, "
            f"ref_code_context_frames={ref_code_context_frames}"
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
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    use_first_chunk = initial_chunk_size > 0 and initial_chunk_size < chunk_size

    if use_first_chunk and length <= initial_chunk_size:
        if not finished and length < initial_chunk_size:
            return None
        context_length = length if finished and length < initial_chunk_size else initial_chunk_size
    else:
        # The initial chunk is only for TTFA. After that, return to the normal
        # codec chunk size so Code2Wav is not flooded by repeated tiny windows.
        initial_coverage = initial_chunk_size if use_first_chunk else 0
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Prepend a bounded ref_code tail as decoder context for every chunk so the
    # vocoder keeps voice-clone speaker identity without making Stage1 shapes
    # depend on full reference-audio length. The decoder is causal with sliding
    # attention, so frames older than this context window cannot affect the
    # emitted chunk. Use `.get()` (not `.pop()`) to keep ref_code for later chunks.
    ref_code = request_payload.get(request_id)
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        ref_context = ref_code
        if ref_code_context_frames > 0 and int(ref_context.shape[0]) > ref_code_context_frames:
            logger.info_once(
                "Qwen3-TTS async chunk uses the last %d/%d ref_code frames as bounded Code2Wav context.",
                ref_code_context_frames,
                int(ref_context.shape[0]),
            )
            ref_context = ref_context[-ref_code_context_frames:]
        ref_frames = ref_context.tolist()
        window_frames = ref_frames + window_frames
        left_context_size += len(ref_frames)

    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)
    code_predictor_codes = torch.tensor(
        [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)],
        dtype=torch.long,
    )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
        speaker=extract_speaker_from_request(request),
        language=extract_language_from_request(request),
    )


# ============================================================================
# Worker-connector data plane (non-async-chunk path).
# AR runner's `flatten_payload` converts the model emit
# `multimodal_outputs={"codes": {"audio": ..., "ref": ...},
# "meta": {"ref_code_len": ..., "codec_streaming": ...}}` to flat dotted
# keys (`codes.audio`, `codes.ref`, `meta.ref_code_len`,
# `meta.codec_streaming`) before the full-payload accumulator runs.
# - codes.audio is 2-D so default CONCAT across steps builds the full sequence.
# - codes.ref is a list (not Tensor with dim>=2) so accumulator LATEST-wins
#   keeps the prefill-emitted ref tensor across decode steps (which don't emit
#   ref again).
# - meta.ref_code_len is 1-D so LATEST-wins; consumer reads [-1].
# ============================================================================

# Per-model REPLACE-keys for the full-payload accumulator.  qwen3_tts's
# producer side emits codec frames that should CONCAT (codes.audio) plus
# scalars/lists that are correctly handled by default LATEST-wins, so this
# stays empty.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()

_CODEBOOK_SIZE = 2048
_NUM_QUANTIZERS_DEFAULT = 16


def _filter_audio_codes_qwen3_tts(audio_codes: torch.Tensor) -> torch.Tensor:
    """Filter zero-padded, out-of-range, and negative-padded codec frames.

    Mirrors the orchestrator-path body in `talker2code2wav` above.
    """
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return audio_codes
    if audio_codes.ndim != 2:
        return audio_codes
    valid_mask = (
        (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
    )
    return audio_codes[valid_mask]


def _coerce_ref_code_len(raw) -> int:
    """Coerce mm["meta"]["ref_code_len"] / pooling_output["meta.ref_code_len"]
    raw value (Tensor | int | None) into a non-negative int.  Mirrors the
    extraction inlined in the legacy ``talker2code2wav`` path; clamps any
    negative input to 0 since downstream code treats this as a non-negative
    frame count."""
    if isinstance(raw, torch.Tensor):
        value = int(raw.reshape(-1)[-1].item()) if raw.numel() > 0 else 0
    elif raw is None:
        value = 0
    else:
        value = int(raw)
    return max(value, 0)


def _normalize_ref_code(ref_code, num_quantizers: int, ref_code_len: int):
    """Coerce ref_code into a [ref_len, Q] tensor or None.  Mirrors orchestrator path."""
    if isinstance(ref_code, list):
        ref_code = ref_code[0] if ref_code else None
    if not isinstance(ref_code, torch.Tensor) or ref_code.numel() == 0:
        return None, 0
    ref_code = ref_code.to(torch.long).cpu().contiguous()
    if ref_code.ndim == 1:
        if ref_code.numel() % num_quantizers != 0:
            return None, 0
        ref_code = ref_code.reshape(-1, num_quantizers)
    elif ref_code.ndim != 2:
        return None, 0
    if ref_code_len > 0 and int(ref_code.shape[0]) > ref_code_len:
        ref_code = ref_code[:ref_code_len]
    return ref_code, int(ref_code.shape[0])


def talker2code2wav_token_only(
    source_outputs: list,
    prompt=None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync-side placeholder for the non-async-chunk Stage-1 (code2wav) input.

    Sized to the expected codec token count (codebook-major flat:
    Q * (ref_frames + audio_frames)).  Speaker / language metadata are
    extracted from `prompt` and threaded via `additional_information`
    (orchestrator-style; same as the legacy `talker2code2wav` builder).
    Actual codec ids are delivered via the worker connector payload built
    by `talker2code2wav_full_payload`.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list = []
    for i, talker_output in enumerate(source_outputs):
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if hasattr(output, "multimodal_output") else None
        mm = mm if isinstance(mm, dict) else {}
        mm_codes = mm.get("codes", {}) if isinstance(mm, dict) else {}
        token_ids = getattr(output, "cumulative_token_ids", []) or []
        seq_len = max(len(token_ids) - 1, 0)

        audio = mm_codes.get("audio") if isinstance(mm_codes, dict) else None
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            audio = audio.to(torch.long)
            audio = _filter_audio_codes_qwen3_tts(audio)
            if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
                audio = audio[-seq_len:]
            num_audio_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
            num_quantizers = int(audio.shape[1]) if audio.ndim == 2 and audio.shape[1] > 0 else _NUM_QUANTIZERS_DEFAULT
        else:
            num_audio_frames = 0
            num_quantizers = _NUM_QUANTIZERS_DEFAULT

        ref_code_raw = mm_codes.get("ref") if isinstance(mm_codes, dict) else None
        ref_code_len_raw = mm.get("meta", {}).get("ref_code_len") if isinstance(mm.get("meta"), dict) else None
        ref_code_len = _coerce_ref_code_len(ref_code_len_raw)
        _, ref_frames = _normalize_ref_code(ref_code_raw, num_quantizers, ref_code_len)

        # Codebook-major flat: Q * (ref_frames + audio_frames)
        prompt_len = num_quantizers * (ref_frames + num_audio_frames)

        additional_info = to_dict(
            OmniPayloadStruct(
                meta=MetaStruct(left_context_size=ref_frames) if ref_frames > 0 else None,
                speaker=extract_speaker_from_prompt(prompt, index=i),
                language=extract_language_from_prompt(prompt, index=i),
            )
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=additional_info if additional_info else None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_full_payload(
    transfer_manager,
    pooling_output,
    request,
):
    """Producer-side payload builder.

    Reads accumulated codec from `pooling_output["codes.audio"]` (CONCAT
    across steps via flatten_payload), latest `pooling_output["codes.ref"]`
    (prefill-emitted), and latest `pooling_output["meta.ref_code_len"]`.
    Replicates the orchestrator-path body of `talker2code2wav` (filter,
    crop to seq_len, prepend ref, codebook-major flatten).
    """
    del transfer_manager
    rid = getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "qwen3_tts.talker2code2wav_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    # codes.audio — try flat dotted first (flatten_payload), then nested fallback.
    audio = pooling_output.get("codes.audio")
    if audio is None:
        codes_nested = pooling_output.get("codes")
        if isinstance(codes_nested, dict):
            audio = codes_nested.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        logger.warning(
            "qwen3_tts.talker2code2wav_full_payload: missing/empty codes.audio "
            "(keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None
    audio = audio.to(torch.long)
    audio = _filter_audio_codes_qwen3_tts(audio)
    if audio.numel() == 0:
        logger.warning(
            "qwen3_tts.talker2code2wav_full_payload: audio empty after codec "
            "filter (negative/all-zero/out-of-range rows dropped) for req=%s.",
            rid,
        )
        return None

    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    seq_len = max(len(output_token_ids) - 1, 0)
    if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
        audio = audio[-seq_len:]

    num_quantizers = int(audio.shape[1]) if audio.ndim == 2 and audio.shape[1] > 0 else _NUM_QUANTIZERS_DEFAULT

    # meta.ref_code_len — flat dotted then nested fallback.
    ref_code_len_raw = pooling_output.get("meta.ref_code_len")
    if ref_code_len_raw is None:
        meta_nested = pooling_output.get("meta")
        if isinstance(meta_nested, dict):
            ref_code_len_raw = meta_nested.get("ref_code_len")
    ref_code_len = _coerce_ref_code_len(ref_code_len_raw)

    # codes.ref — flat dotted then nested fallback.
    ref_code_raw = pooling_output.get("codes.ref")
    if ref_code_raw is None:
        codes_nested = pooling_output.get("codes")
        if isinstance(codes_nested, dict):
            ref_code_raw = codes_nested.get("ref")
    ref_code, ref_frames = _normalize_ref_code(ref_code_raw, num_quantizers, ref_code_len)
    if ref_code is not None:
        audio = torch.cat([ref_code.to(audio.device), audio], dim=0)

    codec_codes = audio.transpose(0, 1).to(device="cpu", dtype=torch.long).reshape(-1).contiguous()
    return {
        "codes": {"audio": codec_codes},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
