# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TTS prompt builder + scope validators for higgs-audio v2.

vllm-omni's higgs path supports two request shapes:

* **Plain text -> 24 kHz speech.** :func:`build_plain_text_prompt` runs the
  upstream HF processor with a bare ``"Generate audio following instruction."``
  system prompt; the emitted ``input_ids`` are byte-identical to upstream.
* **Voice clone (shallow).** :func:`build_voice_clone_prompt` runs the
  upstream HF processor with both a target text turn and a reference
  ``(ref_audio, ref_text)`` ICL turn; the processor returns ``input_ids``
  pre-expanded with audio placeholders plus an ``audio_input_ids`` tensor
  carrying the encoded reference codes (HF tokenizer encodes audio on the
  spot — no encoder vendored in vllm-omni).

Still out of scope and rejected with explicit 4xx: multi-speaker
``[SPEAKERn]`` dialogue, ``profile:`` text-only speaker descriptions, the
``ref_audio_in_system_message`` system-block variant, and chunked long-form
generation.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import torch

__all__ = [
    "UnsupportedInputError",
    "MULTI_SPEAKER_TAG_PATTERN",
    "REJECTED_REQUEST_FIELDS",
    "validate_plain_text_request",
    "validate_plain_text_input",
    "build_plain_text_conversation",
    "build_plain_text_prompt",
    "build_voice_clone_conversation",
    "build_voice_clone_prompt",
    "input_ids_to_python_list",
]


class UnsupportedInputError(ValueError):
    """Raised when a request asks for an out-of-scope higgs_audio_v2 feature."""


# Matches the upstream multi-speaker SPEAKERn tag, e.g. [SPEAKER0], [SPEAKER12].
MULTI_SPEAKER_TAG_PATTERN = re.compile(r"\[SPEAKER\d+\]", re.IGNORECASE)

# Rich-input aliases the validator still rejects so callers using upstream
# field names hit a 4xx pointing at the canonical (``ref_audio``, ``ref_text``)
# spellings instead of silently dropping the audio. ``ref_audio`` and
# ``ref_text`` themselves are now accepted for shallow voice clone.
# ``messages`` + ``speakers`` (multi-speaker dialogue) remain out of scope.
REJECTED_REQUEST_FIELDS: tuple[str, ...] = (
    "reference_audio",
    "voice_prompt",
    "speaker_audio",
    "speakers",
    "messages",
)


def validate_plain_text_input(text: str) -> None:
    """Reject multi-speaker tags inside the user text body.

    Phase-1 explicitly does NOT support multi-speaker dialogue. Catching the
    pattern here means the rejection happens at the tokenizer boundary and is
    visible to both offline (`pipeline.py`) and online (`serving_speech.py`)
    code paths.
    """
    if not isinstance(text, str):
        raise UnsupportedInputError(
            f"higgs_audio_v2 expects plain text input; got {type(text).__name__}"
        )
    if MULTI_SPEAKER_TAG_PATTERN.search(text):
        raise UnsupportedInputError(
            "higgs_audio_v2 v1 does not support multi-speaker [SPEAKERn] tags; "
            "received text contains a speaker tag"
        )


def validate_plain_text_request(request_payload: dict[str, Any]) -> None:
    """Walk through a request dict and reject any out-of-scope field.

    The serving layer calls this BEFORE building the prompt so the 4xx error
    message can name the model and the offending field. Anything still
    present in :data:`REJECTED_REQUEST_FIELDS` after the validator is treated
    as a hard reject regardless of value.
    """
    for field in REJECTED_REQUEST_FIELDS:
        if field in request_payload and request_payload[field] not in (None, "", [], {}):
            raise UnsupportedInputError(
                f"higgs_audio_v2 v1 does not support the request field "
                f"{field!r}; supply plain text via the 'input' field instead"
            )

    text = request_payload.get("input")
    if text is None:
        raise UnsupportedInputError(
            "higgs_audio_v2 requires plain text in the 'input' field"
        )
    validate_plain_text_input(text)


def build_plain_text_conversation(text: str) -> list[dict[str, Any]]:
    """Build the canonical single-speaker plain-text conversation.

    Uses the bare system prompt ``"Generate audio following instruction."``
    that matches the upstream HF reference's input formatting; this exact
    wording is required for input-token parity with the upstream processor.
    """
    validate_plain_text_input(text)
    system_prompt = "Generate audio following instruction."
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]


def build_plain_text_prompt(
    processor: Any,
    text: str,
    *,
    sampling_rate: int = 24000,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Run the upstream processor's chat template on a plain-text input.

    Returns the processor output dict (``input_ids`` plus any auxiliary tensors
    such as ``attention_mask``). The serving layer passes ``input_ids`` to
    Stage 0 as ``prompt_token_ids`` after a ``.tolist()``.

    Using the upstream processor verbatim (no system-prompt rewriting) is
    what preserves input-token parity with the HF reference.
    """
    conversation = build_plain_text_conversation(text)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        sampling_rate=sampling_rate,
        return_tensors=return_tensors,
    )
    if "input_ids" not in inputs:
        raise RuntimeError(
            "HiggsAudioV2 processor returned no input_ids; got keys "
            f"{list(inputs.keys())!r}"
        )
    return inputs


def input_ids_to_python_list(inputs: dict[str, Any]) -> list[int]:
    """Convenience: pull a flat ``list[int]`` of token IDs from a processor output."""
    ids = inputs["input_ids"]
    if isinstance(ids, torch.Tensor):
        if ids.ndim == 2 and int(ids.shape[0]) != 1:
            raise ValueError(
                f"expected batch=1 prompt; got input_ids shape {tuple(ids.shape)}"
            )
        return ids.reshape(-1).tolist()
    return list(ids)


_AUDIO_OUT_TOKEN = "<|AUDIO_OUT|>"
_AUDIO_OUT_BOS_TOKEN = "<|audio_out_bos|>"
_AUDIO_EOS_TOKEN = "<|audio_eos|>"
_AUDIO_DELAY_TOKEN = "<|reserved_special_token_6|>"
_AUDIO_STREAM_BOS_ID = 1024
_AUDIO_STREAM_EOS_ID = 1025


def _build_delay_pattern(codes: torch.Tensor) -> torch.Tensor:
    """Apply the upstream delay-pattern wrap to a ``[num_codebooks, T]`` code tensor.

    Mirrors ``HiggsAudioV2Processor.build_delay_pattern``: prepend BOS, append
    EOS, then arrange in a triangular pattern that stretches the sequence by
    ``num_codebooks - 1`` frames so codebook k starts emitting real codes at
    frame k.
    """
    num_codebooks, seq_len = codes.shape
    bos = codes.new_full((num_codebooks, 1), _AUDIO_STREAM_BOS_ID)
    eos = codes.new_full((num_codebooks, 1), _AUDIO_STREAM_EOS_ID)
    wrapped = torch.cat([bos, codes, eos], dim=1)
    wrapped_len = wrapped.shape[1]
    new_seq_len = wrapped_len + num_codebooks - 1

    output = torch.ones(
        (1, num_codebooks, new_seq_len), dtype=codes.dtype, device=codes.device
    )
    bos_mask = torch.tril(output, -1) > 0
    eos_mask = torch.triu(output, wrapped_len) > 0
    data_mask = ~(bos_mask | eos_mask)
    output[bos_mask] = _AUDIO_STREAM_BOS_ID
    output[data_mask] = wrapped.reshape(-1)
    output[eos_mask] = _AUDIO_STREAM_EOS_ID
    return output[0]


_ENCODER_CACHE: Any | None = None


def _load_upstream_boson_tokenizer(
    tokenizer_id: str = "bosonai/higgs-audio-v2-tokenizer",
):
    """Construct + weight-load the upstream ``HiggsAudioTokenizer`` directly.

    Bypasses ``boson_multimodal.load_higgs_audio_tokenizer`` because the Hub's
    ``config.json`` has been updated to the new HF nested-config schema
    (``acoustic_model_config``, ``semantic_model_config``) that the older
    upstream constructor signature doesn't accept. We pass only the kwargs
    ``HiggsAudioTokenizer.__init__`` actually consumes, mapped from the new
    schema where they differ.
    """
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
        HiggsAudioTokenizer,
    )
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(tokenizer_id)
    config_path = os.path.join(snapshot_path, "config.json")
    weights_path = os.path.join(snapshot_path, "model.pth")
    import json

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    acoustic_cfg = cfg.get("acoustic_model_config") or {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HiggsAudioTokenizer(
        # Map the new HF nested config to the upstream constructor signature.
        n_filters=32,
        D=int(acoustic_cfg.get("hidden_size", 256)),
        target_bandwidths=[1, 1.5, 2, 4, 6],
        # downsampling_ratios product = sample_rate / frame_rate; for the
        # higgs-v2 tokenizer this is [8, 5, 4, 2, 3] = 960 (25 Hz at 24 kHz).
        ratios=list(acoustic_cfg.get("downsampling_ratios", [8, 5, 4, 2, 3])),
        sample_rate=int(cfg.get("sample_rate", 24000)),
        bins=int(cfg.get("codebook_size", 1024)),
        n_q=int(acoustic_cfg.get("n_codebooks", 8)) - 1 if int(acoustic_cfg.get("n_codebooks", 8)) == 9 else 8,
        codebook_dim=int(cfg.get("codebook_dim", 64)),
        semantic_techer="hubert_base_general",
        semantic_sample_rate=int(cfg.get("semantic_sample_rate", 16000)),
        device=device,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Only flag the structural keys; some keys (e.g. semantic_model
        # internals re-initialized from facebook/hubert-base-ls960 via
        # HuggingFace AutoModel.from_pretrained inside the tokenizer constructor)
        # are expected.
        critical_missing = [
            k for k in missing
            if not k.startswith(("semantic_model.", "_codebook.inited"))
        ]
        if critical_missing:
            raise RuntimeError(
                f"upstream higgs audio tokenizer is missing {len(critical_missing)} "
                f"non-semantic-model params: {critical_missing[:5]}"
            )
    return model.to(device).eval()


def _encode_ref_audio_codes(
    wav: "np.ndarray",
    sr: int,
    *,
    tokenizer_id: str = "bosonai/higgs-audio-v2-tokenizer",
) -> torch.Tensor:
    """Encode a single ref clip to codec codes via the upstream boson tokenizer.

    The HF transformers ``HiggsAudioV2TokenizerModel.from_pretrained`` can't
    load the boson-ai ``model.pth`` — the Hub's ``model.safetensors`` contains
    the talker weights, not the encoder/decoder — so the ``audio_tokenizer``
    attached to ``HiggsAudioV2Processor`` runs on randomly-initialized weights
    and produces noise codes. We instead instantiate the upstream
    ``HiggsAudioTokenizer`` directly (see :func:`_load_upstream_boson_tokenizer`),
    which loads ``model.pth`` cleanly.

    Returns shape ``[num_codebooks, T_raw]`` (before BOS/EOS + delay-pattern wrap).
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        try:
            _ENCODER_CACHE = _load_upstream_boson_tokenizer(tokenizer_id)
        except ImportError as exc:
            extra_path = os.environ.get("HIGGS_AUDIO_BOSON_MULTIMODAL_PATH")
            raise RuntimeError(
                "higgs_audio_v2 voice clone needs the upstream "
                "`boson_multimodal` package on PYTHONPATH (the HF transformers "
                "audio_tokenizer model class cannot load boson-ai's model.pth). "
                "Install it via `pip install boson-multimodal` or set "
                "HIGGS_AUDIO_BOSON_MULTIMODAL_PATH to a checked-out copy of "
                "https://github.com/boson-ai/higgs-audio."
                + (f" HIGGS_AUDIO_BOSON_MULTIMODAL_PATH={extra_path!r}" if extra_path else "")
            ) from exc

    # ``HiggsAudioTokenizer.encode`` accepts either a path or a raw waveform;
    # its internal feature extractor resamples to the configured sample rate.
    codes = _ENCODER_CACHE.encode(wav, sr=sr) if sr else _ENCODER_CACHE.encode(wav)
    if isinstance(codes, torch.Tensor):
        return codes.detach().to("cpu").long()
    return torch.as_tensor(codes, dtype=torch.long)


def build_voice_clone_conversation(
    text: str,
    ref_text: str,
) -> list[dict[str, Any]]:
    """ChatML conversation for voice clone (HF jinja-template-compatible).

    The assistant turn is a list with a single ``{"type": "audio"}`` content
    block — no ``"audio"``/``"url"``/``"path"`` key, so ``apply_chat_template``
    won't try to extract audio data (that path would collide with our explicit
    audio encoder). The template renders the assistant block as the literal
    ``<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>``; we expand the single
    ``<|AUDIO_OUT|>`` token to ``N × audio_token + (num_codebooks-1) × delay_token``
    in :func:`build_voice_clone_prompt` to match the reference clip's frame count.
    """
    validate_plain_text_input(text)
    validate_plain_text_input(ref_text)
    return [
        {"role": "system", "content": "Generate audio following instruction."},
        {"role": "user", "content": ref_text},
        {"role": "assistant", "content": [{"type": "audio"}]},
        {"role": "user", "content": text},
    ]


def build_voice_clone_prompt(
    processor: Any,
    text: str,
    ref_audio_wav: "np.ndarray | torch.Tensor",
    ref_audio_sr: int,
    ref_text: str,
    *,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Build a voice-clone prompt using the upstream boson audio tokenizer.

    Returns a dict carrying:
      - ``prompt_token_ids``: ``list[int]`` — input_ids with the ref-audio
        ``<|AUDIO_OUT|>`` + ``<|reserved_special_token_6|>`` placeholders
        already expanded to match the encoded reference clip's frame count.
      - ``audio_input_ids``: ``Tensor[T_frames, num_codebooks]`` — encoded
        reference codes with BOS/EOS + delay pattern, transposed to ``[T, Q]``
        for parity with the HF processor output shape.
      - ``audio_input_ids_mask``: ``Tensor[T_frames]`` — all-True bool mask.
    """
    if isinstance(ref_audio_wav, torch.Tensor):
        wav = ref_audio_wav.detach().to("cpu").float().reshape(-1).numpy()
    else:
        wav = np.asarray(ref_audio_wav, dtype=np.float32).reshape(-1)

    # 1. Encode via the upstream boson tokenizer (real weights from model.pth).
    codes_qt = _encode_ref_audio_codes(wav, ref_audio_sr)  # [num_codebooks, T_raw]
    if codes_qt.ndim == 3:
        codes_qt = codes_qt[0]
    num_codebooks = int(codes_qt.shape[0])

    # 2. Apply BOS/EOS + delay-pattern wrap.
    audio_input_ids = _build_delay_pattern(codes_qt)  # [num_codebooks, T_full]
    T_full = int(audio_input_ids.shape[1])

    # 3. Render the chat template, then expand the single <|AUDIO_OUT|> marker
    #    in the assistant turn into the full placeholder block. After expansion
    #    the text contains exactly T_full audio-mask positions, matching the
    #    audio_input_ids time axis.
    conversation = build_voice_clone_conversation(text, ref_text)
    rendered = processor.tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    n_delay = num_codebooks - 1
    n_audio = T_full - n_delay
    if n_audio < 0:
        raise RuntimeError(
            f"ref clip too short ({T_full} frames) for delay pattern with "
            f"num_codebooks={num_codebooks}"
        )
    placeholders = _AUDIO_OUT_TOKEN * n_audio + _AUDIO_DELAY_TOKEN * n_delay
    expanded_assistant = f"{_AUDIO_OUT_BOS_TOKEN}{placeholders}{_AUDIO_EOS_TOKEN}"
    if _AUDIO_OUT_TOKEN not in rendered:
        raise RuntimeError(
            f"Voice-clone chat-template render is missing the assistant "
            f"audio placeholder marker. conversation={conversation!r}"
        )
    rendered = rendered.replace(_AUDIO_OUT_TOKEN, expanded_assistant, 1)

    # 4. Tokenize the rendered prompt. ``apply_chat_template`` already emits
    #    ``<|begin_of_text|>``, so disable add_special_tokens.
    encoded = processor.tokenizer(
        rendered, add_special_tokens=False, return_tensors=return_tensors,
    )
    prompt_token_ids = encoded["input_ids"]
    if isinstance(prompt_token_ids, torch.Tensor):
        prompt_token_ids = prompt_token_ids.reshape(-1).tolist()

    # 5. Transpose codes to [T_full, num_codebooks] for HF parity.
    audio_input_ids_t = audio_input_ids.transpose(0, 1).contiguous().to(torch.long)
    audio_input_ids_mask = torch.ones(T_full, dtype=torch.bool)

    return {
        "prompt_token_ids": prompt_token_ids,
        "audio_input_ids": audio_input_ids_t,
        "audio_input_ids_mask": audio_input_ids_mask,
    }
