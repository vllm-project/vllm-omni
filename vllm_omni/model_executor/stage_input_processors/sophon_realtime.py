# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage transitions for Sophon-Realtime.

Thinker -> talker: every thinker step, the talker gets the token ids of the reply
written so far, re-tokenized for the talker (cumulative list in ``ids.output``):

* what is spoken is cleaned first: tool calls are never spoken (text up to the
  first ``<tool_call>``), markdown symbols and bracketed asides are dropped, and
  the sound tags the talker was trained on are kept whole
  (``[laughter] [breath] [sigh] [cough] [quick_breath]``,
  ``<laughter>..</laughter>``, ``<strong>..</strong>``);
* the last token is held back while the thinker is writing (it may still merge
  with the next characters); everything is released when the thinker finishes;
* the first chunk waits for a few tokens so the talker does not run dry.

Tag (CosyVoice3 talker) re-tokenizes with the CosyVoice3 tokenizer, where each
sound tag is one token, and gets a chunk on every thinker step once started (its
talker only steps when a chunk arrives and needs several steps per text token).
Its first chunk also carries the voice prompt (``embed``), which the stock
CosyVoice3 talker -> code2wav processor hands to the flow.
Flash (Qwen3-TTS talker) uses the thinker's tokenizer and gets a chunk only when
new text is ready.

Talker -> vocoder uses the stock CosyVoice3 / Qwen3-TTS processors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.data_entry_keys import EmbeddingsStruct, IdsStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.model_executor.models.sophon_realtime.core.text import (  # noqa: F401  (re-exported)
    SOUND_TAG_RE,
    clean_for_speech,
    speakable_text,
    strip_sound_tags,
)
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_request,
    extract_speaker_from_request,
)

TAG_LEAD_TOKENS = 5
FLASH_LEAD_TOKENS = 3


@dataclass
class _StreamState:
    sent: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _states(transfer_manager) -> dict[str, _StreamState]:
    states = getattr(transfer_manager, "_sophon_realtime_streams", None)
    if states is None:
        states = transfer_manager._sophon_realtime_streams = {}
    return states


def _thinker_tokenizer(transfer_manager):
    tok = getattr(transfer_manager, "_sophon_realtime_thinker_tokenizer", None)
    if tok is None:
        model_config = getattr(transfer_manager, "model_config", None) or getattr(transfer_manager, "config", None)
        tok = cached_tokenizer_from_config(model_config)
        if tok is None:
            raise RuntimeError("Sophon-Realtime: the thinker tokenizer is required to stream text to the talker")
        transfer_manager._sophon_realtime_thinker_tokenizer = tok
    return tok


def _talker_dir(transfer_manager) -> str:
    """``<checkpoint>/talker``; the thinker stage is served from ``<checkpoint>/thinker``."""
    model_config = getattr(transfer_manager, "model_config", None) or getattr(transfer_manager, "config", None)
    thinker_dir = str(getattr(model_config, "model", "")).rstrip("/")
    return os.path.join(os.path.dirname(thinker_dir), "talker")


def _cosyvoice3_tokenizer(transfer_manager):
    tok = getattr(transfer_manager, "_sophon_realtime_talker_tokenizer", None)
    if tok is None:
        from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer

        path = os.path.join(_talker_dir(transfer_manager), "CosyVoice-BlankEN")
        tok = get_qwen_tokenizer(path, skip_special_tokens=True, version="cosyvoice3")
        transfer_manager._sophon_realtime_talker_tokenizer = tok
    return tok


def _voice_bank(transfer_manager):
    bank = getattr(transfer_manager, "_sophon_realtime_voices", None)
    if bank is None:
        from vllm_omni.model_executor.models.sophon_realtime.voices import VoiceBank

        talker_dir = _talker_dir(transfer_manager)
        bank = VoiceBank(os.path.join(os.path.dirname(talker_dir), "voices"), talker_dir)
        bank.start_scanning()
        transfer_manager._sophon_realtime_voices = bank
    return bank


def _voice_prompt(transfer_manager, request) -> EmbeddingsStruct:
    """Flow conditioning of the request's voice. A clip that is still being
    prepared is not waited for: the language's default voice is used."""
    lang = extract_language_from_request(request)  # a list of entries
    if isinstance(lang, (list, tuple)):
        lang = lang[0] if lang else None
    lang = str(lang or "zh").lower()[:2]
    name = extract_speaker_from_request(request) or ""
    _, cond = _voice_bank(transfer_manager).load(name, lang, clips=False)
    token = cond["speech_token"]
    return EmbeddingsStruct(
        speech_token=token,
        speech_feat=cond["speech_feat"],
        embedding=cond["embedding"],
        speech_token_len=torch.tensor([[token.shape[1]]], dtype=torch.int32),
    )


def _payload(request, ids: list[int], finished: bool, embed: EmbeddingsStruct | None = None) -> OmniPayloadStruct:
    return OmniPayloadStruct(
        ids=IdsStruct(output=list(ids)),
        meta=MetaStruct(finished=torch.tensor(bool(finished), dtype=torch.bool)),
        speaker=extract_speaker_from_request(request),
        language=extract_language_from_request(request),
        embed=embed,
    )


def _spoken_text(transfer_manager, request, finished: bool) -> str:
    tok = _thinker_tokenizer(transfer_manager)
    out = [int(t) for t in (request.output_token_ids or [])]
    text = tok.decode(out, skip_special_tokens=True)
    return clean_for_speech(speakable_text(text, finished), finished).lstrip()


def thinker2talker_tag_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Thinker -> CosyVoice3 talker (Tag): cumulative CosyVoice3 token ids of the
    spoken text, sent on every thinker step once ``TAG_LEAD_TOKENS`` are ready."""
    rid = request.external_req_id
    states = _states(transfer_manager)
    state = states.setdefault(rid, _StreamState())
    text = _spoken_text(transfer_manager, request, is_finished)
    ids = _cosyvoice3_tokenizer(transfer_manager).encode(text, allowed_special="all") if text else []
    stable = ids if is_finished else ids[:-1]
    if is_finished:
        states.pop(rid, None)
    elif (state.sent == 0 and len(stable) < TAG_LEAD_TOKENS) or len(stable) < state.sent:
        return None
    embed = None
    if not state.extra.get("voice_sent"):
        embed = _voice_prompt(transfer_manager, request)
        state.extra["voice_sent"] = True
    state.sent = max(state.sent, len(stable), 1)
    return _payload(request, stable, is_finished, embed)


def thinker2talker_flash_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Thinker -> Qwen3-TTS talker (Flash): cumulative thinker token ids of the
    spoken text, sent only when new text is ready."""
    rid = request.external_req_id
    states = _states(transfer_manager)
    state = states.setdefault(rid, _StreamState())
    text = _spoken_text(transfer_manager, request, is_finished)
    tok = _thinker_tokenizer(transfer_manager)
    ids = tok(text, add_special_tokens=False).input_ids if text else []
    stable = ids if is_finished else ids[:-1]
    if is_finished:
        states.pop(rid, None)
    elif (state.sent == 0 and len(stable) < FLASH_LEAD_TOKENS) or len(stable) <= state.sent:
        return None
    state.sent = len(stable)
    return _payload(request, stable, is_finished)


# Token-only processors: without this flag the scheduler calls them only when the
# thinker finishes, so the talker would start after the whole reply is written.
thinker2talker_tag_async_chunk.requires_token_updates = True  # type: ignore[attr-defined]
thinker2talker_flash_async_chunk.requires_token_updates = True  # type: ignore[attr-defined]
