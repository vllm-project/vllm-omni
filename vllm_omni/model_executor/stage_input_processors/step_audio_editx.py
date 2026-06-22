"""Stage input processor for Step-Audio-EditX: AR -> Code2Wav."""

from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayload, OmniPayloadStruct, to_dict
from vllm_omni.engine.mm_outputs import MultimodalPayload

logger = init_logger(__name__)

AUDIO_TOKEN_OFFSET = 65536


def _payload_get(payload: Any, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    getter = getattr(payload, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _payload_keys(payload: Any) -> list[str]:
    if payload is None:
        return []
    keys = getattr(payload, "keys", None)
    if callable(keys):
        try:
            return [str(k) for k in keys()]
        except TypeError:
            return []
    return []


def _extract_ref_audio(additional_information: Any) -> Any:
    if isinstance(additional_information, dict):
        return additional_information.get("ref_audio", additional_information.get("latent"))
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("ref_audio") or entries.get("latent")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if list_data is not None:
        return list_data
    scalar_data = getattr(entry, "scalar_data", None)
    if scalar_data is not None:
        return scalar_data
    return None


def _offset_audio_tokens_to_codec(tokens: Any) -> torch.Tensor:
    if tokens is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(tokens, torch.Tensor):
        token_tensor = tokens.detach().to(torch.long).cpu().reshape(-1)
    else:
        token_tensor = torch.tensor([int(token) for token in tokens], dtype=torch.long)
    return token_tensor[token_tensor >= AUDIO_TOKEN_OFFSET] - AUDIO_TOKEN_OFFSET


def _offset_ref_tokens_to_codec(ref_code: torch.Tensor) -> torch.Tensor:
    ref_code = ref_code.to(torch.long).cpu().contiguous()
    if int(ref_code.min().item()) < AUDIO_TOKEN_OFFSET:
        raise RuntimeError("ref_code should be offset by 65536; unexpected StepAudio AR output")
    return ref_code - AUDIO_TOKEN_OFFSET


def _extract_ref_payload(mm: OmniPayload | MultimodalPayload | None) -> torch.Tensor:
    mm_codes = {}
    ref_code = None

    if isinstance(mm, MultimodalPayload):
        metadata = mm.metadata
        mm_codes = metadata.get("codes", {})
        if isinstance(mm_codes, Mapping):
            ref_code = mm_codes.get("ref")
    else:
        mm_codes = _payload_get(mm, "codes", {})
        ref_code = _payload_get(mm_codes, "ref")
        if ref_code is None:
            ref_code = _payload_get(mm, "codes.ref")

    if isinstance(ref_code, list):
        ref_code = ref_code[0] if ref_code else None

    if not isinstance(ref_code, torch.Tensor) or ref_code.numel() == 0:
        raise RuntimeError(
            "StepAudio AR output is missing reference codec tokens "
            f"(codes.ref); multimodal keys={_payload_keys(mm)}, "
            f"codes keys={_payload_keys(mm_codes)}"
        )

    return _offset_ref_tokens_to_codec(ref_code)


def talker2code2wav_token_only(
    source_outputs: list,
    prompt=None,
    _requires_multimodal_data: bool = False,
) -> list:
    """
    Sync path directly builds Code2Wav input
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    additional_information = prompt.get("additional_information", None)
    ref_audio = _extract_ref_audio(additional_information)
    code2wav_inputs: list = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if hasattr(output, "multimodal_output") else None
        audio_codes = output.token_ids if output.token_ids is not None else []
        audio_codes = _offset_audio_tokens_to_codec(audio_codes)
        ref_code = _extract_ref_payload(mm)

        additional_information = to_dict(
            OmniPayloadStruct(
                codes=CodesStruct(ref=ref_code),
            )
        )
        if ref_audio is not None:
            additional_information["ref_audio"] = ref_audio
        audio_codes_len = int(audio_codes.numel())
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * audio_codes_len,
                additional_information=additional_information if additional_information else None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_full_payload(transfer_manager, pooling_output, request):
    del transfer_manager
    audio = pooling_output.get("codes.audio")
    ref_code = pooling_output.get("codes.ref")

    if audio is None or ref_code is None:
        return None

    audio = _offset_audio_tokens_to_codec(audio)
    if audio.numel() == 0:
        return None
    ref_code = _offset_ref_tokens_to_codec(ref_code)

    payload = {
        "codes": {
            "audio": audio,
            "ref": ref_code,
        },
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
        },
    }

    return payload


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    additional_information = getattr(request, "additional_information", None)
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)

    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    state = transfer_manager.request_payload.setdefault(request_id, {})

    if not state.get("has_ref_conditioning", False):
        ref_audio = _extract_ref_audio(additional_information)
        if ref_audio is not None:
            state["pending_ref_audio"] = ref_audio

        ref_code = _extract_ref_payload(multimodal_output)

        if ref_code is not None:
            state["pending_ref_code"] = ref_code
        if state.get("pending_ref_code") is not None and state.get("pending_ref_audio") is not None:
            state["has_ref_conditioning"] = True

    seen_len = int(state.get("seen_len", 0))

    output_token_ids = list(getattr(request, "output_token_ids", []) or [])
    new_tokens = output_token_ids[seen_len:]
    state["seen_len"] = len(output_token_ids)

    for tok in new_tokens:
        tok = int(tok)
        if tok >= AUDIO_TOKEN_OFFSET:
            transfer_manager.code_prompt_token_ids[request_id].append([tok - AUDIO_TOKEN_OFFSET])

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 75))

    sent_audio_len = int(state.get("sent_audio_len", 0))
    audio_tokens = transfer_manager.code_prompt_token_ids[request_id]

    available = len(audio_tokens) - sent_audio_len

    if not finished and available < chunk_size:
        return None

    take = available if finished else chunk_size
    chunk_frames = audio_tokens[sent_audio_len : sent_audio_len + take]

    advance = available if finished else chunk_size
    state["sent_audio_len"] = sent_audio_len + advance

    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        raise RuntimeError(
            "StepAudioEditX AR produced no audio codec tokens "
            f"for request {request_id}; finished={finished}, "
            f"output_token_ids={len(output_token_ids)}"
        )

    code_predictor_codes = torch.tensor(
        [frame[0] for frame in chunk_frames],
        dtype=torch.long,
    )

    has_conditioning = state.get("has_ref_conditioning", False)
    sent_ref_conditioning = state.get("sent_ref_conditioning", False)

    if has_conditioning and not sent_ref_conditioning:
        ref_audio = state.get("pending_ref_audio")
        ref_code = state.get("pending_ref_code")
        state["sent_ref_conditioning"] = True
    else:
        ref_audio = None
        ref_code = None

    if finished:
        transfer_manager.request_payload.pop(request_id, None)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes, ref=ref_code),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=finished,
            req_id=request_id,
        ),
        latent=ref_audio,
    )
